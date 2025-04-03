#pragma once

using GLint = int;
using GLsizei = int;
using GLuint = unsigned int;
using GLenum = unsigned int;

struct FrameBuffer {

    GLuint id = 0;

    GLsizei _width = 0;
    GLsizei _height = 0;
    GLsizei _samples = 1;

    GLint _internalFormatColor = 0;
    GLenum _formatColor = 0;

    GLuint colorTextureId = 0;
    GLuint depthTextureId = 0;

    FrameBuffer(GLsizei width, GLsizei height);
    FrameBuffer(
            GLsizei width,
            GLsizei height, 
            GLsizei samples, 
            GLint internalFormatColor, 
            GLenum formatColor);

    void destroy();

    void resize(GLsizei width, 
            GLsizei height, 
            GLsizei samples = -1, 
            GLint internalFormatColor = 0,
            GLenum formatColor = 0);
};
