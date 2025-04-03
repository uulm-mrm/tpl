#include "tplcpp/dyn_prog/occupancy_renderer.cuh"

#define EGL_EGLEXT_PROTOTYPES
#include <GL/glew.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cuda_gl_interop.h>

#include <cmath>
#include <ostream>
#include <sstream>
#include <iostream>

/*
 * Shaders! Did you know that the GPU can be used for graphics?
 */

const std::string srcVertexShader = R"(
#version 330

uniform float scale;

layout(location = 0) in vec2 in_vert;
layout(location = 1) in uint in_color;
flat out uint color;

void main() {
    float x = in_vert.x * scale;
    float y = in_vert.y * scale;
    gl_Position = vec4(x, y, 0.0, 1.0);
    color = in_color;
}
)";

const std::string srcFragmentShader = R"(
#version 330

flat in uint color;
out uint fragColor;

void main() {
    fragColor = color;
}
)";

OccupancyRenderer::OccupancyRenderer() {

    if (!eglInitialized) {
        initEgl();
    }

    idVertexShader = initGlShader(GL_VERTEX_SHADER, srcVertexShader);
    idFragmentShader = initGlShader(GL_FRAGMENT_SHADER, srcFragmentShader);
    idShaderProgram = initGlShaderProgram();
    initVertexBuffer();

    fb = std::make_unique<FrameBuffer>(1, 1);
}

OccupancyRenderer::~OccupancyRenderer() {

    glDeleteShader(idVertexShader);
    glDeleteShader(idFragmentShader);
    glDeleteProgram(idShaderProgram);
    glDeleteVertexArrays(1, &idVertexArray);
    glDeleteBuffers(1, &idVertexBuffer);
}

void OccupancyRenderer::initEgl() {

    static const int MAX_DEVICES = 32;
    EGLDeviceEXT eglDevs[MAX_DEVICES];
    EGLint numDevices;

    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);
    
    if (0 == numDevices) {
        throw std::runtime_error("Found 0 EGL capable devices!");
    }

    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
          "eglGetPlatformDisplayEXT");

    const char* selectedDevice = std::getenv("CUDA_VISIBLE_DEVICES");
    size_t gpuId = 0;

    if (selectedDevice != nullptr) {
        gpuId = (size_t)std::min(MAX_DEVICES, std::stoi(selectedDevice));
    }

    EGLDisplay eglDpy =
      eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[gpuId], 0);

    if (0 == eglDpy) {
        throw std::runtime_error("EGL display creation has failed!");
    }

    // ok we got our virtual display, initialize

    EGLint major = 0;
    EGLint minor = 0;

    if (!eglInitialize(eglDpy, &major, &minor)) {
        throw std::runtime_error("EGL initialization has failed!");
    }

    // create virtual surface for rendering

    EGLint numConfigs;
    EGLConfig eglCfg;

    const EGLint configAttribs[] = {
          EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
          EGL_BLUE_SIZE, 8,
          EGL_GREEN_SIZE, 8,
          EGL_RED_SIZE, 8,
          EGL_DEPTH_SIZE, 8,
          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
          EGL_NONE
    };

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, 800,
        EGL_HEIGHT, 800,
        EGL_NONE,
    };

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    EGLSurface eglSurf = eglCreatePbufferSurface(
            eglDpy, eglCfg, pbufferAttribs);

    if (0 == eglSurf) {
        throw std::runtime_error("EGL surface creation has failed!");
    }

    eglBindAPI(EGL_OPENGL_API);

    // create opengl context

    EGLContext eglCtx = eglCreateContext(
            eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

    if (0 == eglCtx) {
        throw std::runtime_error("EGL context creation has failed!");
    }

    glewExperimental = true;
    GLenum initResult = glewInit();

    // Complaining about having no GLX display, is ok in EGL mode,
    // as there is (by definition) no X-Server involved.
    if (GLEW_ERROR_NO_GLX_DISPLAY != initResult && GLEW_OK != initResult) {
        throw std::runtime_error("GLEW initialization with EGL has failed!");
    }

    eglInitialized = true;
}

GLuint OccupancyRenderer::initGlShader(GLenum type, std::string source) {

    GLuint id = glCreateShader(type);

	const GLchar* sourceCode = source.c_str();

	glShaderSource(id, 1, &sourceCode, 0);
	glCompileShader(id);

	int compileState;
	glGetShaderiv(id, GL_COMPILE_STATUS, &compileState);

    if (compileState == GL_TRUE) {
        return id;
    }

    int logSize;
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &logSize);

    char* logData = new char[logSize];
    glGetShaderInfoLog(id, logSize, &logSize, logData);

    glDeleteShader(id);

    std::stringstream ss;
    ss << "Shader compilation error: " << std::endl;
    ss << logData << std::endl;

    delete[] logData;

    throw std::runtime_error(ss.str());
}

GLuint OccupancyRenderer::initGlShaderProgram() {

    idShaderProgram = glCreateProgram();

	if (idShaderProgram == 0) {
		std::cout << "Program creation failed!" << std::endl;
		std::cout << "OpenGl error:" << glGetError() << std::endl;
        throw std::runtime_error("Program creation failed!");
	}

    glAttachShader(idShaderProgram, idVertexShader);
	glAttachShader(idShaderProgram, idFragmentShader);

	glLinkProgram(idShaderProgram);

	int linkState;

	glGetProgramiv(idShaderProgram, GL_LINK_STATUS, &linkState);

	if (linkState == GL_TRUE) {
        return idShaderProgram;
    }

    int logSize;
    glGetProgramiv(idShaderProgram, GL_INFO_LOG_LENGTH, &logSize);

    char* logData = new char[logSize];
    glGetProgramInfoLog(idShaderProgram, logSize, &logSize, logData);

    std::stringstream ss;
    ss << "Program linking error:" << std::endl;
    ss << logData << std::endl;
    
    glDeleteProgram(idShaderProgram);

    delete[] logData;

    throw std::runtime_error(ss.str());
}

void OccupancyRenderer::initVertexBuffer() {

    glGenVertexArrays(1, &idVertexArray);
    glBindVertexArray(idVertexArray);

    glGenBuffers(1, &idVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, idVertexBuffer);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vertex),
        0);

    glEnableVertexAttribArray(1);
    glVertexAttribIPointer(
        1,
        1,
        GL_UNSIGNED_INT,
        sizeof(Vertex),
        (GLvoid*)offsetof(Vertex, t));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OccupancyRenderer::checkGlError(const char* msg) {

    GLenum glerr = glGetError();
    if (glerr != GL_NO_ERROR) {
        std::cout << msg << ", gl error code: " << glerr << std::endl;
    }
}

void OccupancyRenderer::render(OccupancyRendererParams& params) {

    occTexture.size = params.s_max + params.l_max*2.0;
    occTexture.size_pixels = (int)(std::ceil(occTexture.size * params.ppm));
    occTexture.ppm = params.ppm;
    
    fb->resize(occTexture.size_pixels,
               occTexture.size_pixels,
               1,
               GL_R32UI,
               GL_RED_INTEGER);

    glEnable(GL_COLOR_LOGIC_OP);
    glLogicOp(GL_OR);

    glUseProgram(idShaderProgram);

    glBindFramebuffer(GL_FRAMEBUFFER, fb->id);
    glViewport(0, 0, fb->_width, fb->_height);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(idVertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, idVertexBuffer);

    GLint scaleLoc = glGetUniformLocation(idShaderProgram, "scale");
    glUniform1f(scaleLoc, 1.0/(occTexture.size*0.5));

    glBufferData(GL_ARRAY_BUFFER,
                 vertices.size()*sizeof(Vertex),
                 vertices.data(),
                 GL_STREAM_DRAW);

    checkGlError("copying occupancy vertex data failed");

    glDrawArrays(GL_TRIANGLES, 0, vertices.size());

    vertices.resize(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glUseProgram(0);

    checkGlError("occupancy rendering failed");
}

void OccupancyRenderer::mapCudaTexture() {

    cudaError_t err{};

    err = cudaGraphicsGLRegisterImage(
            &occRes,
            fb->colorTextureId,
            GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsReadOnly);
    checkCudaError(err, "registering occupancy texture failed");

    err = cudaGraphicsMapResources(1, &occRes);
    checkCudaError(err, "mapping occupancy texture failed");

    cudaArray_t occResArray;
    err = cudaGraphicsSubResourceGetMappedArray(&occResArray, occRes, 0, 0);
    checkCudaError(err, "could not get array access to occupancy texture");

    cudaResourceDesc occTexRes;
    memset(&occTexRes, 0, sizeof(cudaResourceDesc));
    occTexRes.resType = cudaResourceTypeArray;
    occTexRes.res.array.array = occResArray;

    cudaTextureDesc occTexDescr;
    memset(&occTexDescr, 0, sizeof(cudaTextureDesc));
    occTexDescr.normalizedCoords = 0;
    occTexDescr.filterMode = cudaFilterModePoint;
    occTexDescr.addressMode[0] = cudaAddressModeClamp;
    occTexDescr.addressMode[1] = cudaAddressModeClamp;
    occTexDescr.addressMode[2] = cudaAddressModeClamp;
    occTexDescr.readMode = cudaReadModeElementType;

    err = cudaCreateTextureObject(&occTexture.tex,
                                  &occTexRes,
                                  &occTexDescr,
                                  nullptr);
    checkCudaError(err, "cuda occupancy texture creation failed");
}

void OccupancyRenderer::unmapCudaTexture() {

    cudaDestroyTextureObject(occTexture.tex);

    cudaError_t err = cudaGraphicsUnmapResources(1, &occRes);
    checkCudaError(err, "unmapping occupancy texture failed");

    err = cudaGraphicsUnregisterResource(occRes);
    checkCudaError(err, "unregister occupancy texture failed");
}

void OccupancyRenderer::readCartTex(unsigned int* data) {

    // Warning! Incredibly slow!

    glReadPixels(
            0, 0,
            fb->_width, fb->_height,
            fb->_formatColor,
            GL_UNSIGNED_INT,
            data);
}
