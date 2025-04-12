import math
import os
import sys
import moderngl
from PIL import Image
import numpy as np
import glm
import imageio.v3 as iio
from objloader import Obj

class ImageTexture:
    def __init__(self, img):
        self.ctx = moderngl.get_context()
        width, height = img.size
        channels = len(img.getbands())
        self.texture = self.ctx.texture((width, height), channels, img.tobytes())
        self.sampler = self.ctx.sampler(texture=self.texture)

    def use(self, location=0):
        self.sampler.use(location=location)

ctx = moderngl.create_context(standalone=True, backend='egl', require=330)  # Require OpenGL 3.3 core profile

with open('resources/vert.shader', 'r') as f:
    vertex_shader = f.read()

with open('resources/frag.shader', 'r') as f:
    fragment_shader = f.read()

program = ctx.program(
    vertex_shader=vertex_shader,
    fragment_shader=fragment_shader,
)

program['Albedo'].value = 0  # Match location 0 for texture1
program['Roughness'].value = 2  # Match location 1 for texture2
program['Normal'].value = 1  # Match location 1 for texture2
program['hdrTexture'].value = 3  # Match location 1 for texture2
program['Metallic'].value = 4  # Match location 1 for texture2

campos =(0.5,1.0, 1.0)
def camera_matrix():

    proj = glm.perspective(55.0, 1.0, 0.1, 100.0)
    look = glm.lookAt(campos, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    return proj * look

class Model:

    def __init__(self,ctx, program):
        obj = Obj.open("resources/sphere.obj")
        self.vbo = ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))

        self.vao = ctx.vertex_array(program, [
            (self.vbo, '3f 3f 2f', 'in_vertex', 'in_normal', 'in_uv')
        ])

     
cam_matrix = camera_matrix()
program['camera'].write(cam_matrix.to_bytes())
program['cameraPos'].value= campos
program['height_factor'].value= 1

quad= Model(ctx,program) 

# Load HDR image
hdr_image = iio.imread('resources/studio.hdr')

hdr_image = hdr_image.astype(np.float32)/255
height, width, _ = hdr_image.shape
hdrtexture = ctx.texture((width, height), 3, hdr_image.tobytes(), dtype='f4')
hdrsampler = ctx.sampler(texture=hdrtexture)
# Set texture parameters (important for HDR textures)
hdrtexture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)  # Mipmap filter
hdrsampler.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)  # Mipmap filter
hdrtexture.build_mipmaps()
hdrsampler.use(location=3)

rtexture = ctx.texture((512, 512), 4)  # RGBA texture
depthbuffer = ctx.depth_renderbuffer((512, 512))
fbo = ctx.framebuffer(color_attachments=[rtexture], depth_attachment=depthbuffer)
# enable depht test
ctx.enable(moderngl.DEPTH_TEST)
fbo.use()

#        img = Image.open(path).convert('RGBA')

def render(img_albedo, img_normal, img_roughness, img_metallic):

    texture = ImageTexture(img_albedo)
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)  # Mipmap filter
    texture.use(0)

    texturen = ImageTexture(img_normal)
    texturen.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)  # Mipmap filter
    texturen.use(1)

    texturer = ImageTexture(img_roughness)
    texturer.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)  # Mipmap filter
    texturer.use(2)

    texturerm = ImageTexture(img_metallic)
    texturerm.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)  # Mipmap filter
    texturerm.use(4)

    ctx.clear(0.0, 0.0, 0.0, 1.0)  # Clear framebuffer with opaque black
    quad.vao.render()

    # Save texture to file
    data = rtexture.read()
    return Image.frombytes('RGBA', rtexture.size, data)

