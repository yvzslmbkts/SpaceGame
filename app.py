import sys

import glm
import numpy as np
from OpenGL import GL as gl, GLUT as glut
from OpenGL.GL import shaders
from SkyboxShader import SkyboxShader
from Utilities import *
from Camera import *
import time
from copy import deepcopy
import random
import TextureLoader


class GameObject:
    def __init__(self, obj_data: BoundObjData, parent: 'GameObject' = None, position=glm.vec3(0.0), scale=glm.vec3(1.0), rotation=glm.quat(glm.vec3(0.0))):
        self.obj_data = obj_data

        self.parent = parent

        self.position = position
        self.rotation = rotation
        self.scale = scale

        self.__transformation_matrix = None
        self.__transformations_changed = True

    def get_position(self):
        return self.__position

    def set_position(self, position):
        self.__position = position
        self.__transformations_changed = True

    position = property(get_position, set_position)

    def get_rotation(self):
        return self.__rotation

    def set_rotation(self, rotation):
        self.__rotation = rotation
        self.__transformations_changed = True

    rotation = property(get_rotation, set_rotation)

    def get_scale(self):
        return self.__scale

    def set_scale(self, scale):
        self.__scale = scale
        self.__transformations_changed = True

    scale = property(get_scale, set_scale)

    def get_final_position(self):
        return (self.get_transformation() * glm.vec4(0.0, 0.0, 0.0, 1.0)).xyz

    def get_final_scale(self):
        return self.scale if self.parent is None else self.scale * self.parent.get_final_scale()

    def get_final_rotation(self):
        return self.rotation if self.parent is None else self.rotation * self.parent.get_final_rotation()

    def get_transformation(self):
        if self.__transformations_changed:
            transformation = glm.mat4x4()
            transformation = glm.translate(transformation, self.position)
            transformation = transformation * glm.mat4_cast(self.rotation)
            transformation = glm.scale(transformation, self.scale)
            self.__transformation_matrix = transformation
            self.__transformations_changed = False

        if self.parent is None:
            return self.__transformation_matrix
        else:
            return self.parent.get_transformation() * self.__transformation_matrix

    def draw(self,type):
        # Set the vao
        gl.glBindVertexArray(self.obj_data.vao)
        self.texture_type = type
        #Set material uniforms
        
        # gl.glUniform3fv(ambient_color_location, 1, glm.value_ptr(self.obj_data.meshes[0].material.Ka))
        # gl.glUniform3fv(diffuse_color_location, 1, glm.value_ptr(self.obj_data.meshes[0].material.Kd))
        # gl.glUniform3fv(specular_color_location, 1, glm.value_ptr(self.obj_data.meshes[0].material.Ks))
        # gl.glUniform1fv(shininess_location, 1, self.obj_data.meshes[0].material.Ns)

        # Set material textures
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D,type)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D,type)

        # Set the transform uniform to self.transform
        gl.glUniformMatrix4fv(transformation_location, 1, False, glm.value_ptr(self.get_transformation()))

        # Draw the mesh with an element buffer.
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.obj_data.meshes[0].element_array_buffer)
        # When the last parameter in 'None', the buffer bound to the GL_ELEMENT_ARRAY_BUFFER will be used.
        gl.glDrawElements(gl.GL_TRIANGLES, self.obj_data.meshes[0].element_count, gl.GL_UNSIGNED_INT, None)

    def get_AABB(self):
        dimensions = self.obj_data.dimensions
        scale = self.get_final_scale()
        position = self.get_final_position()

        return tuple(
            (dimensions[axis][0] * scale[axis] + position[axis], dimensions[axis][1] * scale[axis] + position[axis])
            for axis in range(3)
        )

    def check_AABB_collision(self, AABB):
        self_AABB = self.get_AABB()

        return (self_AABB[0][0] <= AABB[0][1] and self_AABB[0][1] >= AABB[0][0])\
            and (self_AABB[1][0] <= AABB[1][1] and self_AABB[1][1] >= AABB[1][0])\
            and (self_AABB[2][0] <= AABB[2][1] and self_AABB[2][1] >= AABB[2][0])

    def drawAABB(self):
        # Set the polygon mode to lines so we can see the actual object inside the AABB
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        # Set the vao
        gl.glBindVertexArray(primitive_objs["cube"].vao)

        # Set material uniforms
        gl.glUniform3fv(ambient_color_location, 1, glm.value_ptr(self.obj_data.meshes[0].material.Ka))
        gl.glUniform3fv(diffuse_color_location, 1, glm.value_ptr(self.obj_data.meshes[0].material.Kd))
        gl.glUniform3fv(specular_color_location, 1, glm.value_ptr(self.obj_data.meshes[0].material.Ks))
        gl.glUniform1fv(shininess_location, 1, self.obj_data.meshes[0].material.Ns)

        # Set material textures
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.obj_data.meshes[0].material.map_Ka)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.obj_data.meshes[0].material.map_Kd)

        AABB = self.get_AABB()
        center = glm.vec3(
            (AABB[0][0] + AABB[0][1]) / 2.0,
            (AABB[1][0] + AABB[1][1]) / 2.0,
            (AABB[2][0] + AABB[2][1]) / 2.0
        )

        scale = glm.vec3(
            (AABB[0][1] - AABB[0][0]) / 2.0,
            (AABB[1][1] - AABB[1][0]) / 2.0,
            (AABB[2][1] - AABB[2][0]) / 2.0
        )

        AABB_transformation = glm.mat4x4()
        AABB_transformation = glm.translate(AABB_transformation, center)
        AABB_transformation = glm.scale(AABB_transformation, scale)

        # Set the transform uniform to self.transform
        gl.glUniformMatrix4fv(transformation_location, 1, False, glm.value_ptr(AABB_transformation))

        # Draw the mesh with an element buffer.
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, primitive_objs["cube"].meshes[0].element_array_buffer)
        # When the last parameter in 'None', the buffer bound to the GL_ELEMENT_ARRAY_BUFFER will be used.
        gl.glDrawElements(gl.GL_TRIANGLES, primitive_objs["cube"].meshes[0].element_count, gl.GL_UNSIGNED_INT, None)

        # Set the state to its initial
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)





# Initialize GLUT ------------------------------------------------------------------|
glut.glutInit()
glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)

# Create a window
screen_size = glm.vec2(800, 600)
glut.glutCreateWindow("Space Game")
glut.glutReshapeWindow(int(screen_size.x), int(screen_size.y))


skybox_texture = TextureLoader.load_images_to_cubemap_texture(
    "Assets/skybox_bg/morningdew_rt.tga", "Assets/skybox_bg/morningdew_lf.tga",
    "Assets/skybox_bg/morningdew_up.png", "Assets/skybox_bg/morningdew_dn.png",
    "Assets/skybox_bg/morningdew_bk.tga", "Assets/skybox_bg/morningdew_ft.tga",
)

lives = 100
hit_recovery_time = 2.0
last_hit_at = 0.0
time_passed = 0.0
score = 0

# Set callback functions
def display():
    global time_passed
    delta_time = time.perf_counter() - time_passed
    time_passed += delta_time
    # Clear screen
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)


    # gl.glDepthMask(gl.GL_FALSE)
    # gl.glCullFace(gl.GL_FRONT)
    # skybox_shader.draw(
    #     perspective_projection,
    #     glm.mat4x4(glm.mat3x3(camera.get_view_matrix())),
    #     # glm.mat4x4(),
    #     skybox_texture
    # )
    # gl.glDepthMask(gl.GL_TRUE)
    # gl.glCullFace(gl.GL_BACK)


    gl.glUniformMatrix4fv(projection_location, 1, False, glm.value_ptr(perspective_projection))

    gl.glUniformMatrix4fv(view_location, 1, False, glm.value_ptr(camera.get_view_matrix()))

    gl.glUniform3fv(camera_position_location, 1, glm.value_ptr(camera.camera_pos))

    # Demonstrate some collision checking
    # for collision_tester in collision_testers:
    #     if main_object.check_AABB_collision(collision_tester.get_AABB()):
    #         collision_tester.obj_data.meshes[0].material.Kd = glm.vec3(0.0, 0.0, 1.0)
    #         collision_tester.obj_data.meshes[0].material.Ka = glm.vec3(0.0, 0.0, 1.0)
    #         print('Carptiniz')
    #     else:
    #         collision_tester.obj_data.meshes[0].material.Kd = primitive_objs["sphere"].meshes[0].material.Kd
    #         collision_tester.obj_data.meshes[0].material.Ka = primitive_objs["sphere"].meshes[0].material.Ka

    #collision_tester_parent.rotation = glm.quat(glm.vec3(0.0, 0.0, glm.radians(time.perf_counter() * 10.0)))
    counter = 0
    for collision_tester in collision_testers:
        if (counter % 3 == 0):
            collision_tester.draw(brick)
            collision_tester.rotation = glm.quat(glm.vec3(0.0, 0.0, glm.radians(time.perf_counter() * 20.0)))
        elif (counter % 3 == 1):
            collision_tester.draw(crate)
            collision_tester.rotation = glm.quat(glm.vec3(glm.radians(time.perf_counter() * 20.0),0.0, 0.0))
        elif (counter % 3 == 2):
            collision_tester.draw(metal)
            collision_tester.rotation = glm.quat(glm.vec3(0.0, glm.radians(time.perf_counter() * 20.0) ,0.0))


        #collision_tester.drawAABB()
        counter +=1

    #main_object.drawAABB()
    main_object.position = camera.camera_pos + (0.0, -1.0, -5.0)
    health_object.position = camera.camera_pos + (0.09, 0.06, -0.2)


    global last_hit_at, lives ,textType
    if time_passed >= last_hit_at + hit_recovery_time:
        for collision_tester in collision_testers:
            if main_object.check_AABB_collision(collision_tester.get_AABB()):
                if collision_tester.texture_type == metal:
                    last_hit_at = time_passed
                    lives -= 15
                    textType = 'metal'
                    print(lives)
                elif collision_tester.texture_type == brick:
                    last_hit_at = time_passed
                    lives -= 10
                    textType = 'brick'
                    print(lives)
                elif collision_tester.texture_type == crate:                    
                    last_hit_at = time_passed
                    lives -= 5
                    textType = 'crate'
                    print(lives)
    if (lives <= 0 ):
        sys.exit()
    elif (75 < lives <=100):
        health_object.draw(space_ship)
        main_object.draw(space_ship)
    elif (50 < lives <= 75):
        health_object.draw(space_ship_green)
        main_object.draw(space_ship)
    elif (25 < lives <=50):
        main_object.draw(space_ship)
        health_object.draw(space_ship_yellow)
    elif (0 < lives <= 25):
        main_object.draw(space_ship)
        health_object.draw(space_ship_red)




    if time_passed < last_hit_at + hit_recovery_time:
        if textType == 'metal':
            camera.process_keyboard("FORWARD", 0.3) # This is for moving along the x axis
        elif textType == 'brick':
            camera.process_keyboard("FORWARD", 0.5) # This is for moving along the x axis
        elif textType == 'crate':
            camera.process_keyboard("FORWARD", 0.7) # This is for moving along the x axis
    else:
        camera.process_keyboard("FORWARD", 1)

    score = int(time_passed)
    print(score)
    print(camera.camera_pos)
        


    # Swap the buffer we just drew on with the one showing on the screen
    glut.glutSwapBuffers()

glut.glutDisplayFunc(display)
glut.glutIdleFunc(display)


def resize(width, height):
    gl.glViewport(0, 0, width, height)
    screen_size.x = width
    screen_size.y = height
    global perspective_projection
    perspective_projection = glm.perspective(glm.radians(45.0), screen_size.x / screen_size.y, 0.1, 100.0)

glut.glutReshapeFunc(resize)


def keyboard_input(key, x, y):
    if key == b'\x1b':
        sys.exit()
    if key == b'\x77':
        camera.process_keyboard("UPWARD", 0.4)
    if key == b'\x73':
        camera.process_keyboard("DOWNWARD", 0.4)
    if key == b'\x61':
        camera.process_keyboard("LEFT", 0.4)
    if key == b'\x64':
        camera.process_keyboard("RIGHT", 0.4)

glut.glutKeyboardFunc(keyboard_input)


# def mouse_passive_motion(x, y):
#     mouse_pos = glm.vec2(x, y)

#     # Because the screen dimensions and OpenGL dimensions do not match, mouse position has to be mapped
#     # The mapping is from (0->screen_size.x, 0->screen_size.y) to (-1->1, -1->1)
#     normalized_mouse_pos = (mouse_pos / screen_size) * 2.0 - 1.0
#     # Also flip the y value because screen's bottom is positive direction while OpenGL's top is positive direction
#     normalized_mouse_pos.y *= -1.0
#     # print(normalized_mouse_pos)

#     # Take the inverse of the transformation used to map world coordinates into screen coordinates
#     world2screen_transformation = perspective_projection * camera.get_view_matrix()
#     screen2world_transformation = glm.inverse(world2screen_transformation)

#     # Because I want to transform the mouse position to a plane in the world coordinates where z=0 (just for this case),
#     # I have to find the right value of z to use while creating a 4D point from the 2D mouse position. This will ensure
#     # that the world position
#     p = world2screen_transformation * glm.vec4(random.random(), random.random(), 0.0, 1.0)
#     p /= p.w
#     z_value = p.z

#     # Multiply normalized_mouse_pos with it
#     mouse_pos_in_world = screen2world_transformation * glm.vec4(normalized_mouse_pos, z_value, 1.0)
#     mouse_pos_in_world /= mouse_pos_in_world.w

#     main_object.position = mouse_pos_in_world.xyz

# glut.glutPassiveMotionFunc(mouse_passive_motion)


# Creating a Shader Program -------------------------------------------------------|
# Compile shaders [shorthand]
vertex_shader = shaders.compileShader("""#version 330
layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec2 vertex_texture_position;
layout(location = 2) in vec3 vertex_normal;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 transformation;

out vec2 texture_position;
out vec3 normal;
out vec3 world_position;

void main()
{
    //  Because the transformation matrix is 4x4, we have to construct a vec4 from position, so we an multiply them
    vec4 pos = vec4(vertex_position, 1.0);
    gl_Position = projection * view * transformation * pos;

    texture_position = vertex_texture_position;

    vec4 transformed_normal = transpose(inverse(transformation)) * vec4(vertex_normal, 0.0);
    normal = normalize(transformed_normal.xyz);
    
    world_position = (transformation * pos).xyz;
}
""", gl.GL_VERTEX_SHADER)

fragment_shader = shaders.compileShader("""#version 420
layout(binding = 0) uniform sampler2D ambient_color_sampler;
layout(binding = 1) uniform sampler2D diffuse_color_sampler;
in vec2 texture_position;
in vec3 normal;
in vec3 world_position;
uniform vec3 light_position;
uniform vec3 camera_position;

uniform vec3 ambient_color;
uniform vec3 diffuse_color;
uniform vec3 specular_color;
uniform float shininess;

out vec4 fragment_color;

void main()
{
    float ambient_intensity = 0.1;
    
    vec3 color = (ambient_color + texture(ambient_color_sampler, texture_position).xyz) * ambient_intensity;

    
    float diffuse_intensity = 0.9;
    float light_intensity = dot(
        normalize(light_position - world_position),
        normal
    );
    
    color += (diffuse_color + texture(diffuse_color_sampler, texture_position).xyz) * diffuse_intensity * max(0.0, light_intensity);
    
    
    if (light_intensity > 0.0)
    {
        float specular_intensity = 1.0;
        vec3 light_reflection = reflect(
            normalize(world_position - light_position),
            normal
        );
        float reflection_intensity = dot(
            normalize(camera_position - world_position),
            light_reflection
        );
        
        color += specular_color * specular_intensity * max(0.0, pow(max(0.0, reflection_intensity), shininess));
    }
    
    fragment_color = vec4(color, 1.0);
}
""", gl.GL_FRAGMENT_SHADER)

# Compile the program [shorthand]
shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

# Set the program we just created as the one in use
gl.glUseProgram(shader_program)


# Get the transformation location
transformation_location = gl.glGetUniformLocation(shader_program, "transformation")

# Get the view location
view_location = gl.glGetUniformLocation(shader_program, "view")

# Get the projection location
projection_location = gl.glGetUniformLocation(shader_program, "projection")

# Set the light_position uniform
light_position = glm.vec3(0.0, 0.0, 20.0)
light_position_location = gl.glGetUniformLocation(shader_program, "light_position")
gl.glUniform3fv(light_position_location, 1, glm.value_ptr(light_position))

# Get the camera_position location
camera_position_location = gl.glGetUniformLocation(shader_program, "camera_position")

# Get the material uniforms' locations
ambient_color_location = gl.glGetUniformLocation(shader_program, "ambient_color")
diffuse_color_location = gl.glGetUniformLocation(shader_program, "diffuse_color")
specular_color_location = gl.glGetUniformLocation(shader_program, "specular_color")
shininess_location = gl.glGetUniformLocation(shader_program, "shininess")


# Configure GL -----------------------------------------------------------------------|

# Enable depth test
gl.glEnable(gl.GL_DEPTH_TEST)
# Accept fragment if it closer to the camera than the former one
gl.glDepthFunc(gl.GL_LESS)

# This command is necessary in our case to load different type of image formats.
# Read more on https://www.khronos.org/opengl/wiki/Common_Mistakes under "Texture upload and pixel reads"
gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)


# Creating Data Buffers -----------------------------------------------------------|

# With the ability of .obj file loading, all the data will be read from the files and bound to proper buffers
primitive_objs = {
    "plane": parse_and_bind_obj_file("Assets/Primitives/plane.obj"),
    "cube": parse_and_bind_obj_file("Assets/Primitives/cube.obj"),
    "cylinder": parse_and_bind_obj_file("Assets/Primitives/cylinder.obj"),
    "cone": parse_and_bind_obj_file("Assets/Primitives/cone.obj"),
    "sphere": parse_and_bind_obj_file("Assets/Primitives/sphere.obj"),
    "disc": parse_and_bind_obj_file("Assets/Primitives/disc.obj"),
}


crate = TextureLoader.load_texture("res/crate.jpg")
metal = TextureLoader.load_texture("res/metal.jpg")
brick = TextureLoader.load_texture("res/brick.jpg")
space_ship = TextureLoader.load_texture("res/spaceship.jpg")
space_ship_yellow = TextureLoader.load_texture("res/spaceship_yellow.jpg")
space_ship_green = TextureLoader.load_texture("res/spaceship_green.jpg")
space_ship_red = TextureLoader.load_texture("res/spaceship_red.jpg")

world = TextureLoader.load_texture("res/world.jpg")

# Create Camera and Game Objects -----------------------------------------------------|

#skybox_shader = SkyboxShader(deepcopy(primitive_objs["cube"]))

perspective_projection = glm.perspective(glm.radians(45.0), screen_size.x / screen_size.y, 0.1, 100.0)



camera = Camera(position=glm.vec3(0.0, 0.0, 20.0))


collision_tester_parent = GameObject(None)

# Lets create cones orbiting around the origin and give each of them unique mesh data so we can modify them easily
radius = 6.0
count = 1000
collision_testers = []
for i in range(count):
    angle = glm.radians(360.0 * i / count)
    collision_testers.append(
        GameObject(
            deepcopy(primitive_objs["cube"]),
            position=glm.vec3(random.uniform(-2.0,2.0), random.uniform(-2.0,2.0), - i/5 ) * radius,
            parent=collision_tester_parent
        )
    )

# Lets create a game object to demonstrate dynamic parent changes
red_sphere_obj = deepcopy(primitive_objs["sphere"])
red_sphere_obj2 = deepcopy(primitive_objs["sphere"])

#red_sphere_obj.meshes[0].material.Kd = glm.vec3(0.003, 0.003, 0.003)
#red_sphere_obj.meshes[0].material.Ka = glm.vec3(0.003, 0.003, 0.003)
main_object = GameObject(
    red_sphere_obj,
    scale=glm.vec3(0.5),
)

health_object = GameObject(
    red_sphere_obj2,
    scale=glm.vec3(0.01),
)


#Create shader for Skybox // Sikinti burdan cikiyor derstten sonra halledilecek. 

# Start the main loop
glut.glutMainLoop()
