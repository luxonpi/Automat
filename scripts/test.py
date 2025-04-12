from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import sys
import ntpath
import time
import util
import ast
from PIL import Image, ImageDraw, ImageFont
from subprocess import Popen, PIPE
import torch
import renderer as renderer
from datetime import datetime
import torchvision.transforms as transforms
import networks.unet as unet


# Check command-line arguments
if len(sys.argv) != 3:
    print("Usage: python run_model.py <image.jpg> <[hm,hv,rm,rv,mm,mv]>")
    sys.exit(1)

imagepath = sys.argv[1]

try:
    extra_values = ast.literal_eval(sys.argv[2])
    if not isinstance(extra_values, list) or len(extra_values) != 6:
        raise ValueError
    extra_tensor = torch.tensor(extra_values, dtype=torch.float32)
except (ValueError, SyntaxError):
    print("Error: The second argument must be a list of 6 numeric values, e.g., '[0.1, 0.5, 1.2, -0.3, 0.8, 2.0]'.")
    sys.exit(1)

save_path = os.path.join('/var/tmp/ge69fuv/', 'gnet.pth')


model = unet.Unet(9,6,64).to('cuda')  # Replace 'MyModel' with your actual model class

# Load the state dictionary (weights)
state_dict = torch.load(save_path, map_location='cuda')
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("module.", "")  # Remove "module." prefix
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)

# Set model to evaluation mode
model.eval()


# Load and preprocess the image
image = Image.open(imagepath).convert("RGB")
image_t= transforms.functional.resize(image, (256, 256))
image_t= transforms.functional.to_tensor(image_t)
image_t=  transforms.functional.normalize(image_t, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

metallic_mean_t = torch.full((1, image_t.shape[1], image_t.shape[2]), extra_values[4])
metallic_variance_t = torch.full((1, image_t.shape[1], image_t.shape[2]), extra_values[5])

roughness_mean_t = torch.full((1, image_t.shape[1], image_t.shape[2]), extra_values[2])
roughness_variance_t = torch.full((1, image_t.shape[1], image_t.shape[2]), extra_values[3])

height_mean_t = torch.full((1, image_t.shape[1], image_t.shape[2]), extra_values[0])
height_variance_t = torch.full((1, image_t.shape[1], image_t.shape[2]), extra_values[1])

input = torch.cat((image_t, metallic_mean_t, metallic_variance_t, roughness_mean_t, roughness_variance_t,height_mean_t,height_variance_t), 0)

# add batch dimension
input = input.unsqueeze(0)
input = input.to('cuda')

# meta data tensor from extra data 
metadata_tensor = torch.tensor(extra_values, dtype=torch.float32).unsqueeze(0).to('cuda')

# Run the image through the model
with torch.no_grad():
    pbr = model(input,metadata_tensor)

# the output is a batched tensor, remove the batch dimmenions and split the 6 channel image lie this
# channel 0-2: normalmap (RGB)
# channel 3: roughness
# channel 4: metallic
# channel 5: height

image_normal=util.tensor2im(pbr[:,0:3,:,:])
image_roughness=util.tensor2im(pbr[:,3,:,:].repeat(1,3, 1, 1) )
image_height=util.tensor2im(pbr[:,4,:,:].repeat(1,3, 1, 1) )
image_metallic=util.tensor2im(pbr[:,5,:,:].repeat(1,3, 1, 1) )

#img_albedo, img_normal, img_roughness, img_metallic
rendered_r = renderer.render(image, Image.fromarray(image_normal),Image.fromarray(image_roughness),Image.fromarray(image_metallic)) 

#save all images in the test folder use the og name from the input + gen
rendered_r.save(os.path.join("/var/tmp/ge69fuv/results/test", f'{ntpath.basename(imagepath)}_gen.png'))
Image.fromarray(image_normal).save(os.path.join("/var/tmp/ge69fuv/results/test", f'{ntpath.basename(imagepath)}_gen_normal.png'))
Image.fromarray(image_roughness).save(os.path.join("/var/tmp/ge69fuv/results/test", f'{ntpath.basename(imagepath)}_gen_roughness.png'))
Image.fromarray(image_height).save(os.path.join("/var/tmp/ge69fuv/results/test", f'{ntpath.basename(imagepath)}_gen_height.png'))
Image.fromarray(image_metallic).save(os.path.join("/var/tmp/ge69fuv/results/test", f'{ntpath.basename(imagepath)}_gen_metallic.png'))
image.save(os.path.join("/var/tmp/ge69fuv/results/test", f'{ntpath.basename(imagepath)}_albedo.png'))


# Print output
print("PBR Generated")
