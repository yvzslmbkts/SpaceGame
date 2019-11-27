from OpenGL.GL import *
from PIL import Image
import numpy


def load_texture(path):
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open(path)
    img_data = numpy.array(list(image.getdata()), numpy.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return texture

def load_images_to_cubemap_texture(
        right_image_file_name, left_image_file_name,
        top_image_file_name, bottom_image_file_name,
        back_image_file_name, front_image_file_name,
        mag_filter=GL_LINEAR, min_filter=GL_LINEAR_MIPMAP_LINEAR
):
    # Create a texture
    texture = glGenTextures(1)

    # Bind to it
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture)

    first_cubemap_texture_position = GL_TEXTURE_CUBE_MAP_POSITIVE_X

    for image_file_name in (
            right_image_file_name, left_image_file_name,
            top_image_file_name, bottom_image_file_name,
            back_image_file_name, front_image_file_name,
    ):
        # Read image data
        # The y-coordinates of image data and what OpenGL is expecting are reversed. So we flip the y-coordinates
        image = Image.open(image_file_name)#.transpose(Image.FLIP_TOP_BOTTOM)
        # Load the data into a numpy array so we can pass it to OpenGL
        image_data = numpy.array(image.getdata(), dtype=numpy.uint8)

        # Check if the texture has an Alpha (transparency) chanel or not. It won't work properly with the wrong type.
        if image.mode == "RGBA":
            image_type = GL_RGBA
        else:
            image_type = GL_RGB
        # Send the image data to be used as texture
        glTexImage2D(first_cubemap_texture_position, 0, GL_RGBA, image.size[0], image.size[1], 0, image_type, GL_UNSIGNED_BYTE, image_data)

        # Don't forget to close the image (i.e. free the memory)
        image.close()

        first_cubemap_texture_position += 1

    # Generate mipmaps for the texture
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    # Set its parameters
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, mag_filter)
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, min_filter)

    # return the texture id
    return texture
