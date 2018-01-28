#!/usr/bin/python2

import sys
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.GLU import *
import numpy as np

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

def init():
    if not (glColorPointer and glVertexPointer and glDrawElements):
        print("Error! No vertex array support(?)")
        sys.exit(-1)

    if not glfw.init():
        sys.exit(-1)
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "AR stuff", None,
                                None)
    if not window:
        glfw.terminate()
        sys.exit(-1)

    glfw.make_context_current(window)

    wWidth, wHeight = glfw.get_framebuffer_size(window)

    glViewport(0, 0, wWidth, wHeight)
    glClearColor(0, 1.0, 0, 1.0)
    glEnable(GL_DEPTH_TEST)

    return window

'''
def buildShader(**kwargs):
    shader_parts = []
    for name, path in kwargs.items():
        with open(path, 'r') as source:
            shader_parts.append(shaders.compileShader(source.read(), eval(
                                              "GL_%s_SHADER" % name.upper())))
    return shaders.compileProgram(*shader_parts)
'''

def buildShader(**kwargs):
    shader_parts = []
    for name, path in kwargs.items():
        shader_type = eval("GL_%s_SHADER" % name.upper())
        shader = glCreateShader(shader_type)
        with open(path, 'r') as source:
            #glShaderSource(shader, 1, source.read(), None)
            glShaderSource(shader, source.read(), None)
        glCompileShader(shader)
        shader_parts.append(shader)

    shaderProgram = glCreateProgram()
    for shader in shader_parts:
        glAttachShader(shaderProgram, shader)
    glLinkProgram(shaderProgram)

    for shader in shader_parts:
        glDeleteShader(shader)

    return shaderProgram
       

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv:
            print(arg)

    window = init()
    shader = buildShader(vertex="basic.v.glsl", fragment="basic.f.glsl")
    
    ## Begin VAO creation
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vBuffer)
    vertices = np.array([[-0.5, -0.5, 0.0], [0.5,-0.5, 0.0], [0.0,-0.5, 0.0]], dtype='f')
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    '''
    iBuffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iBuffer)
    indices = np.array([[0,1,2]], dtype=np.int32)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    '''

    glBindVertexArray(0)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        # Render here w/ pyopengl
        glUseProgram(shader)

        glBindVertexArray(vao)

        #glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, None)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glBindVertexArray(0)

        glUseProgram(0)
  
        print(glGetError())
        glfw.swap_buffers(window)

        glfw.poll_events()

    glfw.terminate()
