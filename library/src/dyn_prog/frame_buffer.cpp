#include "tplcpp/dyn_prog/frame_buffer.hpp"

#include <GL/glew.h>
#include <stdexcept>

void checkError(const char* msg) {

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        throw std::runtime_error(
                std::string("opengl error code: ") 
                + std::to_string(err) 
                + ", " 
                + std::string(msg));
    }
}

FrameBuffer::FrameBuffer(GLsizei width, GLsizei height) 
    : FrameBuffer(width, height, 1, GL_RGBA, GL_RGBA) {
}

FrameBuffer::FrameBuffer(
        GLsizei width, 
        GLsizei height, 
        GLsizei samples, 
        GLint internalFormatColor, 
        GLenum formatColor) {

    glGenFramebuffers(1, &id);

    resize(width, height, samples, internalFormatColor, formatColor);

    glBindFramebuffer(GL_FRAMEBUFFER, id);

    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("A framebuffer was not completly initialized!");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameBuffer::destroy() {

    glDeleteFramebuffers(1, &id);
    glDeleteTextures(1, &colorTextureId);
    glDeleteTextures(1, &depthTextureId);
}

void FrameBuffer::resize(
        GLsizei width, 
        GLsizei height, 
        GLsizei samples, 
        GLint internalFormatColor,
        GLenum formatColor) {

    if (samples < 1) {
        samples = _samples;
    }
    if (0 == internalFormatColor) {
        internalFormatColor = _internalFormatColor;
    }
    if (0 == formatColor) {
        formatColor = _formatColor;
    }

    // only resize if absolutely necessary

    if (_width == width
            && _height == height
            && _samples == samples
            && _internalFormatColor == internalFormatColor
            && _formatColor == formatColor) { 
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, id);

    // color texture

    if (colorTextureId != 0) {
        glDeleteTextures(1, &colorTextureId);
    } 

    glGenTextures(1, &colorTextureId);

    if (samples > 1) {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, colorTextureId);

        glTexImage2DMultisample(
                     GL_TEXTURE_2D_MULTISAMPLE,
                     samples,
                     internalFormatColor,
                     width,
                     height,
                     false);

        glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D_MULTISAMPLE,
                colorTextureId,
                0);
    } else {
        glBindTexture(GL_TEXTURE_2D, colorTextureId);

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     internalFormatColor,
                     width,
                     height,
                     0,
                     formatColor,
                     GL_UNSIGNED_INT,
                     0);

        checkError("could not create frame buffer texture");

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

        glFramebufferTexture2D(
                GL_FRAMEBUFFER, 
                GL_COLOR_ATTACHMENT0, 
                GL_TEXTURE_2D, 
                colorTextureId,
                0);
    }

    // depth texture

    if (depthTextureId != 0) {
        glDeleteTextures(1, &depthTextureId);
    }

    glGenTextures(1, &depthTextureId);

    if (samples > 1) {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, depthTextureId);

        glTexImage2DMultisample(
                     GL_TEXTURE_2D_MULTISAMPLE,
                     samples,
                     GL_DEPTH24_STENCIL8,
                     width,
                     height,
                     false);

        glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_DEPTH_STENCIL_ATTACHMENT,
                GL_TEXTURE_2D_MULTISAMPLE,
                depthTextureId,
                0);
    } else {
        glBindTexture(GL_TEXTURE_2D, depthTextureId);

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_DEPTH24_STENCIL8,
                     width,
                     height,
                     0,
                     GL_DEPTH_STENCIL,
                     GL_UNSIGNED_INT_24_8,
                     0);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

        glFramebufferTexture(
                GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, depthTextureId, 0);
    }

    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    _width = width;
    _height = height;
    _samples = samples;
    _internalFormatColor = internalFormatColor;
    _formatColor = formatColor;
}
