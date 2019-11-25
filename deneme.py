import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy
from pyrr import matrix44, Vector3
import TextureLoader
from Camera import Camera
from Utilities import *
import random 
from PIL import Image
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys


def loadskybox():
    im1 = Image.open("res/world.jpg").rotate(180).transpose(Image.FLIP_LEFT_RIGHT).resize((512,512))
    texture1 = im1.tostring()

    glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, 1)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE,texture1)

def drawskybox():
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_DEPTH_TEST)
    glColor3f(1,1,1) # front face
    glBindTexture(GL_TEXTURE_2D, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex3f(-10.0, -10.0, -10.0)
    glTexCoord2f(1, 0)
    glVertex3f(10.0, -10.0, -10.0)
    glTexCoord2f(1, 1)
    glVertex3f(10.0, 10.0, -10.0)
    glTexCoord2f(0, 1)
    
    glVertex3f(-10.0, 10.0, -10.0)
    glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)
    glEnable(GL_DEPTH_TEST)


drawskybox()