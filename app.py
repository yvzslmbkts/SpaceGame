import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy
from pyrr import matrix44, Vector3
import TextureLoader
from Camera import Camera
from Utilities import *
import random 



def window_resize(window, width, height):
    glViewport(0, 0, width, height)

cam = Camera()
keys = [False] * 1024
lastX, lastY = 640, 360
first_mouse = True


def key_callback(window, key, scancode, action, mode):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key >= 0 and key < 1024:
        if action == glfw.PRESS:
            keys[key] = True
        elif action == glfw.RELEASE:
            keys[key] = False


def do_movement():
    
    if keys[glfw.KEY_W]:
        cam.process_keyboard("UPWARD", 0.5)
    if keys[glfw.KEY_S]:
        cam.process_keyboard("DOWNWARD", 0.5)
    
    if keys[glfw.KEY_A]:
        cam.process_keyboard("LEFT", 0.5)
    if keys[glfw.KEY_D]:
        cam.process_keyboard("RIGHT", 0.5)


def mouse_callback(window, xpos, ypos):
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    cam.process_mouse_movement(xoffset, yoffset)


def main():
    # initialize glfw
    if not glfw.init():
        return
    counter = 0
    w_width, w_height = 1280, 720
    aspect_ratio = w_width / w_height

    window = glfw.create_window(w_width, w_height, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)
    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)

    #        positions        texture_coords
    cube = [-0.5, -0.5, 0.5, 0.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 0.0,
            0.5, 0.5, 0.5, 1.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 1.0,

            -0.5, -0.5, -0.5, 0.0, 0.0,
            0.5, -0.5, -0.5, 1.0, 0.0,
            0.5, 0.5, -0.5, 1.0, 1.0,
            -0.5, 0.5, -0.5, 0.0, 1.0,

            0.5, -0.5, -0.5, 0.0, 0.0,
            0.5, 0.5, -0.5, 1.0, 0.0,
            0.5, 0.5, 0.5, 1.0, 1.0,
            0.5, -0.5, 0.5, 0.0, 1.0,

            -0.5, 0.5, -0.5, 0.0, 0.0,
            -0.5, -0.5, -0.5, 1.0, 0.0,
            -0.5, -0.5, 0.5, 1.0, 1.0,
            -0.5, 0.5, 0.5, 0.0, 1.0,

            -0.5, -0.5, -0.5, 0.0, 0.0,
            0.5, -0.5, -0.5, 1.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 1.0,
            -0.5, -0.5, 0.5, 0.0, 1.0,

            0.5, 0.5, -0.5, 0.0, 0.0,
            -0.5, 0.5, -0.5, 1.0, 0.0,
            -0.5, 0.5, 0.5, 1.0, 1.0,
            0.5, 0.5, 0.5, 0.0, 1.0]

    cube = numpy.array(cube, dtype=numpy.float32)

    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               8, 9, 10, 10, 11, 8,
               12, 13, 14, 14, 15, 12,
               16, 17, 18, 18, 19, 16,
               20, 21, 22, 22, 23, 20]

    indices = numpy.array(indices, dtype=numpy.uint32)

    vertex_shader = """
    #version 330
    in layout(location = 0) vec3 position;
    in layout(location = 1) vec2 texture_cords;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 proj;

    out vec2 textures;

    void main()
    {
        gl_Position =  proj * view * model * vec4(position, 1.0f);
        textures = texture_cords;
    }
    """

    fragment_shader = """
    #version 330
    in vec2 textures;

    out vec4 color;
    uniform sampler2D tex_sampler;

    void main()
    {
        color = texture(tex_sampler, textures);
    }
    """
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    # position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 5, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # textures
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, cube.itemsize * 5, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)




    """

    instance_array = numpy.array(instance_array, numpy.float32).flatten()


    instanceVBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO)
    glBufferData(GL_ARRAY_BUFFER, instance_array.itemsize * len(instance_array), instance_array, GL_STATIC_DRAW)

    """


    crate = TextureLoader.load_texture("res/crate.jpg")
    metal = TextureLoader.load_texture("res/metal.jpg")
    brick = TextureLoader.load_texture("res/brick.jpg")
    space_ship = TextureLoader.load_texture("res/spaceship.jpg")
    world = TextureLoader.load_texture("res/world.jpg")
     

    glUseProgram(shader)

    glClearColor(0.2, 0.3, 0.2, 1.0)
    glEnable(GL_DEPTH_TEST)

    projection = matrix44.create_perspective_projection_matrix(45.0, aspect_ratio, 0.1, 100.0)

    model_loc = glGetUniformLocation(shader, "model")   
    main_model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "proj")

    cube_positions = [(0.0, 4.0, 0.0), (0.0, 0.0, -4.0), (0.0, 0.0, -7.0), (0.0, 0.0, -10.0),(0.0, -6.0, 0.0),(0.0, 6.0, 0.0)]
    
    main_cube = (0.0, -1.5 ,-5.0)
    
    """ (X,Y,Z) first one is for right end left if + in the right , and if - in the left """
    """second one for y axis """
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


    while not glfw.window_should_close(window):
        
        glfw.poll_events()
        do_movement()

        """For rotating cubes"""
        time = glfw.get_time()

        rot_x = matrix44.create_from_x_rotation(time * 2)
        rot_y = matrix44.create_from_y_rotation(time * 0.5)
        rot_z = matrix44.create_from_z_rotation(time)
        """"""


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = cam.get_view_matrix()
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        counter -=1
        if(counter % 2 == 0):
            cube_positions.append((random.randrange(-10,10) ,random.randrange(-10,10) ,counter))

        glDepthMask(GL_FALSE)
        bg_model = matrix44.create_from_scale((2,2,5))
        glUniformMatrix4fv(main_model_loc, 1, GL_FALSE, matrix44.inverse(view) * bg_model)
        glBindTexture(GL_TEXTURE_2D, world)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glDepthMask(GL_LESS)

        main_model = matrix44.create_from_translation(main_cube)
        """"This is for rotating"""
        rotX = matrix44.multiply(rot_x, main_model)
        glUniformMatrix4fv(main_model_loc, 1, GL_FALSE, matrix44.inverse(view) * main_model * rotX )

        glBindTexture(GL_TEXTURE_2D, space_ship)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        
        
        for i in range(len(cube_positions)):
        
            model = matrix44.create_from_translation(cube_positions[i])
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            
            if i % 3 == 0:
                glBindTexture(GL_TEXTURE_2D, crate)

                """"This is for rotating"""
                rotX = matrix44.multiply(rot_x, model)
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, rotX)
                """"""
            elif i % 3 == 1:
                glBindTexture(GL_TEXTURE_2D, brick)
                """"This is for rotating"""

                rotY = matrix44.multiply(rot_y, model)
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, rotY)
                """"""
                
            else:
                glBindTexture(GL_TEXTURE_2D, metal)
                """"This is for rotating"""

                rotZ = matrix44.multiply(rot_z, model)
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, rotZ)
                """""" 

            glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
            cam.process_keyboard("FORWARD", 0.004) # This is for moving along the x axis

            
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()