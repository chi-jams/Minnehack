"""
simpleglfw.py

Simple Python OpenGL program that uses PyOpenGL + GLFW to get an 
OpenGL 3.2 context.

Author: Mahesh Venkitachalam
"""

import OpenGL
from OpenGL.GL import *

import math, sys, os
import numpy as np
import glutils

import glfw
import cv2

cap = cv2.VideoCapture(0)
REFERENCE_IMG = "cubemap-colored.jpg"
MIN_MATCH_CT = 10

def resizeMat(mat):
    out_mat = np.identity(4)
    for i in range(3):
        for j in range(3):
            out_mat[i][j] = mat[i][j]
    out_mat[3][3] = 1
    print(out_mat)
    return out_mat
    

class Scene:    
    """ OpenGL 3D scene class"""
    # initialization
    def __init__(self):
        # create shader
        vShader = ""
        with open("shaders/spinny.v.glsl", 'r') as src:
            vShader = src.read()
        fShader = ""
        with open("shaders/spinny.f.glsl", 'r') as src:
            fShader = src.read()

        self.program = glutils.loadShaders(vShader, fShader)
      
        self.bkpgm = glutils.loadShaders(*[open("shaders/bkgnd.%s.glsl"%i, "r").read() for i in "vf"])
        self.backTex = glGetUniformLocation(self.bkpgm, b'tex2D')
        self.bkId = glutils.loadTexture('star.png')
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)
        height, width, _ = frame.shape
        glBindTexture(GL_TEXTURE_2D, self.bkId)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame)
        glBindTexture(GL_TEXTURE_2D, 0)

        glUseProgram(self.program)

        self.pMatrixUniform = glGetUniformLocation(self.program, 
                                                   b'uPMatrix')
        self.mvMatrixUniform = glGetUniformLocation(self.program, 
                                                  b'uMVMatrix')
        # texture 
        self.tex2D = glGetUniformLocation(self.program, b'tex2D')
        
        # define triange strip vertices 
        vertexData = np.array(
            [-0.5, 0.5, 0.5, 
              0.5, 0.5, 0.5, 
              -0.5, -0.5, 0.5,
              0.5, -0.5, 0.5,
              0.5, -0.5, -0.5,
              0.5, 0.5, 0.5,
              0.5,0.5,-0.5,
              -0.5,0.5,0.5,
              -0.5,0.5,-0.5,
              -0.5,-0.5,0.5,
              -0.5,-0.5,-0.5,
              0.5,-0.5,-0.5,
              -0.5,0.5,-0.5,
              0.5,0.5,-0.5], np.float32)

        # set up vertex array object (VAO)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # vertices
        self.vertexBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
        # set buffer data 
        glBufferData(GL_ARRAY_BUFFER, 14*len(vertexData), vertexData, 
                     GL_STATIC_DRAW)
        # enable vertex array
        glEnableVertexAttribArray(0)
        # set buffer data pointer
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        # unbind VAO
        glBindVertexArray(0)

        # time
        self.t = 0 

        # texture
        self.texId = glutils.loadTexture('star.png')
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        glBindTexture(GL_TEXTURE_2D, self.texId)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame)
        glBindTexture(GL_TEXTURE_2D, 0)

        # show circle?
        self.showCircle = False
        
        ## opencv stuff
        self.sift = cv2.xfeatures2d.SIFT_create()
        ref_img = cv2.imread(REFERENCE_IMG, 0) 
        self.ref_kp, self.ref_des = self.sift.detectAndCompute(ref_img, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    # step
    def step(self):
        # increment angle
        self.t = (self.t + 1) % 360
        # set shader angle in radians
        #glUniform1f(glGetUniformLocation(self.program, 'uTheta'), 
        #            math.radians(self.t))

    def findFeature(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        tkp, tdes = self.sift.detectAndCompute(grey, None)
        matches = self.flann.knnMatch(self.ref_des, tdes, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_CT:
            src_pts = np.float32([ self.ref_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ tkp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            print(M)
            
            return M

            #h, w = self.ref_img
        
    # render 
    def render(self, pMatrix, mvMatrix):        
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)
        height, width, _ = frame.shape
        transformMat = self.findFeature(frame)
        if transformMat is not None:
            transformMat = resizeMat(transformMat)
            print(transformMat.shape)
        glBindTexture(GL_TEXTURE_2D, self.texId)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame)
        glBindTexture(GL_TEXTURE_2D, self.bkId)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame)
        glBindTexture(GL_TEXTURE_2D, 0)
         # Draw camera footage
        glUseProgram(self.bkpgm)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bkId)
        glUniform1i(self.backTex, 0)

        glBindVertexArray(self.vao)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        glBindVertexArray(0)

        if transformMat is not None:
            # use shader
            glUseProgram(self.program)
            
            # set proj matrix
            glUniformMatrix4fv(self.pMatrixUniform, 1, GL_FALSE, pMatrix)

            # set modelview matrix
            outMat = np.matmul(transformMat, mvMatrix)
            glUniformMatrix4fv(self.mvMatrixUniform, 1, GL_FALSE, outMat)
            #glUniformMatrix4fv(self.mvMatrixUniform, 1, GL_FALSE, mvMatrix)

            # show circle? 
            glUniform1i(glGetUniformLocation(self.program, b'showCircle'), 
                        self.showCircle)

            # enable texture 
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texId)
            glUniform1i(self.tex2D, 0)

            # bind VAO
            glBindVertexArray(self.vao)
            # draw 
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 14)
            # unbind VAO
            glBindVertexArray(0)


class RenderWindow:
    """GLFW Rendering window class"""
    def __init__(self):

        # save current working directory
        cwd = os.getcwd()

        # initialize glfw - this changes cwd
        glfw.glfwInit()
        
        # restore cwd
        os.chdir(cwd)

        # version hints
        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MAJOR, 4)
        glfw.glfwWindowHint(glfw.GLFW_CONTEXT_VERSION_MINOR, 0)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.glfwWindowHint(glfw.GLFW_OPENGL_PROFILE, 
                            glfw.GLFW_OPENGL_CORE_PROFILE)
    
        # make a window
        self.width, self.height = 640, 480
        self.aspect = self.width/float(self.height)
        self.win = glfw.glfwCreateWindow(self.width, self.height, 
                                         b'simpleglfw')
        # make context current
        glfw.glfwMakeContextCurrent(self.win)
        
        # initialize GL
        glViewport(0, 0, self.width, self.height)
        #glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_NEVER)
        glClearColor(0.5, 0.5, 0.5,1.0)

        # set window callbacks
        glfw.glfwSetMouseButtonCallback(self.win, self.onMouseButton)
        glfw.glfwSetKeyCallback(self.win, self.onKeyboard)
        glfw.glfwSetWindowSizeCallback(self.win, self.onSize)        

        # create 3D
        self.scene = Scene()

        # exit flag
        self.exitNow = False

        
    def onMouseButton(self, win, button, action, mods):
        #print 'mouse button: ', win, button, action, mods
        pass

    def onKeyboard(self, win, key, scancode, action, mods):
        #print 'keyboard: ', win, key, scancode, action, mods
        if action == glfw.GLFW_PRESS:
            # ESC to quit
            if key == glfw.GLFW_KEY_ESCAPE: 
                self.exitNow = True
            else:
                # toggle cut
                self.scene.showCircle = not self.scene.showCircle 
        
    def onSize(self, win, width, height):
        #print 'onsize: ', win, width, height
        self.width = width
        self.height = height
        self.aspect = width/float(height)
        glViewport(0, 0, self.width, self.height)

    def run(self):
        # initializer timer
        glfw.glfwSetTime(0)
        t = 0.0
        while not glfw.glfwWindowShouldClose(self.win) and not self.exitNow:
            # update every x seconds
            currT = glfw.glfwGetTime()
            if currT - t > 0.1:
                # update time
                t = currT
                # clear
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
                # build projection matrix
                pMatrix = glutils.perspective(45.0, self.aspect, 0.1, 100.0)
                
                mvMatrix = glutils.lookAt([3.0*math.sin(glfw.glfwGetTime()), 0.0, 3.0*math.cos(glfw.glfwGetTime())], [0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0])
                # render
                self.scene.render(pMatrix, mvMatrix)
                # step 
                self.scene.step()

                glfw.glfwSwapBuffers(self.win)
                # Poll for and process events
                glfw.glfwPollEvents()
        # end
        glfw.glfwTerminate()

    def step(self):
        # clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # build projection matrix
        pMatrix = glutils.perspective(45.0, self.aspect, 0.1, 100.0)
                
        mvMatrix = glutils.lookAt([0.0, 0.0, -2.0], [0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0])
        # render
        self.scene.render(pMatrix, mvMatrix)
        # step 
        self.scene.step()

        glfw.SwapBuffers(self.win)
        # Poll for and process events
        glfw.PollEvents()

# main() function
def main():
    print("Starting simpleglfw. "
          "Press any key to toggle cut. Press ESC to quit.")
    rw = RenderWindow()
    rw.run()

# call main
if __name__ == '__main__':
    main()
