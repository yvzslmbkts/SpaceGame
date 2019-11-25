from collections import OrderedDict
from typing import List, Dict

from OpenGL import GL as gl
import glm
import numpy as np
from PIL import Image


def load_image_to_texture(image_name):
    # This command is necessary in our case to load textures.
    # Read more on https://www.khronos.org/opengl/wiki/Common_Mistakes under "Texture upload and pixel reads"
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

    # Read image data
    # The y-coordinates of image data and what OpenGL is expecting are reversed. So we flip the y-coordinates
    image = Image.open(image_name).transpose(Image.FLIP_TOP_BOTTOM)
    # Load the data into a numpy array so we can pass it to OpenGL
    image_data = np.array(image.getdata(), dtype=np.uint8)

    # Create a texture
    texture = gl.glGenTextures(1)

    # Bind to it
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

    # Set its parameters
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)

    # Check if the texture has an Alpha (transparency) chanel or not. It won't work properly with the wrong type.
    if image.mode == "RGBA":
        image_type = gl.GL_RGBA
    else:
        image_type = gl.GL_RGB
    # Send the image data to be used as texture
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.size[0], image.size[1], 0, image_type, gl.GL_UNSIGNED_BYTE, image_data)

    # Generate mipmaps for the texture
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

    # Don't forget to close the image (i.e. free the memory)
    image.close()

    # return the texture id
    return texture


class Material:
    def __init__(self, name):
        self.properties = dict()
        self.name = name
        self.Ka = glm.vec3(0.0)
        self.Kd = glm.vec3(0.0)
        self.Ks = glm.vec3(0.0)
        self.Ns = 0.0
        self.map_Kd = None


def parse_material_file(material_file_name):
    with open(material_file_name, "r") as material_file:
        materials: Dict[str, Material] = dict()
        current_material: Material = None

        for line in material_file.readlines():
            tokens = line.strip().split()

            if tokens[0] == "newmtl":
                new_material = Material(name=tokens[1])
                materials[tokens[1]] = new_material
                current_material = new_material

            elif tokens[0] == "Ka":
                current_material.Ka = glm.vec3(tuple(map(float, tokens[1:])))

            elif tokens[0] == "Kd":
                current_material.Kd = glm.vec3(tuple(map(float, tokens[1:])))

            elif tokens[0] == "Ks":
                current_material.Ks = glm.vec3(tuple(map(float, tokens[1:])))

            elif tokens[0] == "Ns":
                current_material.Ns = float(tokens[1])

            elif tokens[0] == "map_Kd":
                current_material.map_Kd = tokens[1]

            else:
                current_material.properties[tokens[0]] = tokens[1]

    return materials


class Mesh:
    def __init__(self, name, material, vertex_indices):
        self.name = name
        self.material = material
        self.vertex_indices = vertex_indices


class ParsedObjData:
    def __init__(self, positions, texture_positions, normals, meshes):
        self.positions = positions
        self.texture_positions = texture_positions
        self.normals = normals
        self.meshes: List[Mesh] = meshes


def parse_obj_file(obj_name) -> ParsedObjData:
    if not obj_name.endswith(".obj"):
        raise RuntimeError(f"Can't load '{obj_name}', because it is not an .obj file")

    positions = []
    texture_positions = []
    normals = []

    faces = []

    meshes: List[Mesh] = []

    materials: Dict[str, Material] = dict()
    current_material = None

    current_mesh_name = "unnamed"

    with open(obj_name, "r") as obj_file:
        for line in obj_file.readlines():
            tokens = line.strip().split()

            if len(tokens) == 0 or tokens[0] == "#":
                continue

            elif tokens[0] == "o" or tokens[0] == "g":
                current_mesh_name = "unnamed" if len(tokens) == 1 else tokens[1]

            elif tokens[0] == "v":
                positions.append(tuple(map(float, tokens[1:])))

            elif tokens[0] == "vt":
                texture_positions.append(tuple(map(float, tokens[1:])))

            elif tokens[0] == "vn":
                normals.append(tuple(map(float, tokens[1:])))

            elif tokens[0] == "f":
                vertices = []
                for vertex in tokens[1:]:
                    vertices.append(tuple(map(int, vertex.split("/"))))

                for i in range(2, len(vertices)):
                    faces.append(vertices[0])
                    faces.append(vertices[i - 1])
                    faces.append(vertices[i])

            elif tokens[0] == "usemtl":
                if current_material is None:
                    current_material = materials[tokens[1]]

                else:
                    meshes.append(
                        Mesh(
                            current_mesh_name,
                            current_material,
                            faces[:]
                        )
                    )

                    current_material = materials[tokens[1]]
                    faces = []

            elif tokens[0] == "mtllib":
                materials.update(parse_material_file(tokens[1]))

    # Create a mesh from remaining faces
    meshes.append(
        Mesh(
            current_mesh_name,
            current_material,
            faces[:]
        )
    )

    return ParsedObjData(positions, texture_positions, normals, meshes)


def format_for_draw_elements(parsed_obj_data) -> ParsedObjData:
    unique_vertices = OrderedDict()
    index = 0

    for mesh in parsed_obj_data.meshes:
        for vertex in mesh.vertex_indices:
            if vertex in unique_vertices:
                continue

            unique_vertices[vertex] = index
            index += 1

    positions = []
    texture_positions = []
    normals = []

    # Create position, texture_position and normal arrays in the order
    for vertex in unique_vertices.keys():
        positions.append(parsed_obj_data.positions[vertex[0] - 1])
        texture_positions.append(parsed_obj_data.texture_positions[vertex[1] - 1])
        normals.append(parsed_obj_data.normals[vertex[2] - 1])

    meshes = []

    # Create meshes with element indexing
    for mesh in parsed_obj_data.meshes:
        formatted_indices = []
        for vertex in mesh.vertex_indices:
            formatted_indices.append(unique_vertices[vertex])

        meshes.append(
            Mesh(mesh.name, mesh.material, formatted_indices)
        )

    return ParsedObjData(positions, texture_positions, normals, meshes)


def send_data_to_vertex_buffer(data, location) -> int:
    # Turn data into a np array so we can send it to a GPU buffer
    vertex_data = np.array(
        data,
        dtype=np.float32
    )
    # Get properties of the vertex data
    data_count = vertex_data.shape[1]       # the count of floats per vertex data
    data_stride = vertex_data.strides[0]    # bytes to skip for the next vertex data
    data_offset = gl.ctypes.c_void_p(0)     # beginning offset

    # Request a buffer slot from GPU
    vertex_buffer = gl.glGenBuffers(1)

    # Describe how the position attribute will parse this buffer
    # 1. First, tell where the data will be read from
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer)
    # 2. Second, tell how the data should be read and where to be sent
    gl.glVertexAttribPointer(location, data_count, gl.GL_FLOAT, False, data_stride, data_offset)

    # Enable the attribute
    gl.glEnableVertexAttribArray(location)

    # Send vertex data (which is on CPU memory) to vertex buffer (which is on GPU memory)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)

    # Bind the GL_ARRAY_BUFFER to 0 (default) to keep the state clear
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    return vertex_buffer


def send_indices_to_element_buffer(data) -> int:
    # Turn data into a np array so we can send it to a GPU buffer
    element_data = np.array(
        data,
        dtype=np.uint32
    )

    element_buffer = gl.glGenBuffers(1)

    # Unlike other buffers, it is bound to the GL_ELEMENT_ARRAY_BUFFER
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, element_buffer)

    # Send element data (which is on CPU memory) to element buffer (which is on GPU memory)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, element_data.nbytes, element_data, gl.GL_STATIC_DRAW)

    # Bind GL_ELEMENT_ARRAY_BUFFER to 0 (default) to keep the state clear
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    return element_buffer


def normalize_positions(vertex_data):
    data = np.array(vertex_data, dtype=np.float32)

    data_max = data.max()
    data_min = data.min()

    return (data - data_min) / (data_max - data_min) * 2.0 - 1.0


class BoundMesh:
    def __init__(self, name, material, element_array_buffer, element_array_count):
        self.name = name
        self.material = material
        self.element_array_buffer = element_array_buffer
        self.element_count = element_array_count


def bind_material_textures(material: Material):
    if material.map_Kd is not None:
        material.map_Kd = load_image_to_texture(material.map_Kd)
    else:
        material.map_Kd = 0

    return material


class BoundObjData:
    def __init__(self, positions_buffer, texture_positions_buffer, normals_buffer, bound_meshes, vao=0):
        self.vao = vao
        self.positions_buffer = positions_buffer
        self.texture_positions_buffer = texture_positions_buffer
        self.normals_buffer = normals_buffer
        self.bound_meshes: List[BoundMesh] = bound_meshes


def bind_mesh_data(mesh_data_in_draw_elements_format) -> BoundObjData:
    # Create the position buffer. It will store 3D position of each vertex at location 0
    position_buffer = send_data_to_vertex_buffer(
        mesh_data_in_draw_elements_format.positions,
        0
    )

    # Create the texture positions buffer. It will store 2D texture position of each vertex at location 1
    texture_positions_buffer = send_data_to_vertex_buffer(
        mesh_data_in_draw_elements_format.texture_positions,
        1
    )

    # Create the normals buffer. It will store 3D normal vector of each vertex at location 2
    normals_buffer = send_data_to_vertex_buffer(
        mesh_data_in_draw_elements_format.normals,
        2
    )

    # Create the element buffers for each mesh
    bound_meshes = []

    for mesh in mesh_data_in_draw_elements_format.meshes:
        bound_meshes.append(
            BoundMesh(
                mesh.name,
                bind_material_textures(mesh.material),
                send_indices_to_element_buffer(
                    mesh.vertex_indices
                ),
                len(mesh.vertex_indices)
            )
        )

    return BoundObjData(position_buffer, texture_positions_buffer, normals_buffer, bound_meshes)


def bind_mesh_data_with_vao(mesh_data_in_draw_elements_format):
    # Create and bind a Vertex Array Object to save the Vertex Buffer configurations we will make
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    # Bind buffers
    bound_obj_data = bind_mesh_data(mesh_data_in_draw_elements_format)
    bound_obj_data.vao = vao

    # Bind Vertex Array to 0 (default) to keep the state clear
    gl.glBindVertexArray(0)

    return bound_obj_data
