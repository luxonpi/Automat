import sys
import os
from PIL import Image

# Get the absolute path of the directory containing renderer.py
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Automat/scripts'))

# Add this directory to sys.path
sys.path.append(script_dir)

# Now you can import renderer
import renderer as renderer

visual_eval_index = [5,7,11,18,19,24,28,33,35,45,47,48,51,54,62,70,79,82,83,86,89,95,103,104,112,119,127,128,148,160,161,162,164,165,167,168,172,176,178,189,194,196,204,206,210,212,218,219,226,244,249,253,259,265,266,276,282,294,302,303,324,326,330,347,355,366,377,382,387,407,408,414,415,426,431,437,458,463,470,476,477,497,538,550,551,555,556,582]

evalIndex = 5

# img_albedo = Image.open(f"Datasets/full512/eval/{evalIndex}_albedo.png")
# img_normal = Image.open(f"Datasets/full512/eval/{evalIndex}_normal.png")
# img_roughness =  Image.open(f"Datasets/full512/eval/{evalIndex}_roughness.png")
# img_metallic = Image.open(f"Datasets/full512/eval/{evalIndex}_metallic.png")

# img = renderer.render(img_albedo,img_normal,img_roughness,img_metallic)
# img.save(f"Results/Render/Image_{evalIndex}.png")  # Save the combined image#

img_albedo = Image.open(f"Resources/chess_inverted.png")
img_normal = image = Image.new("RGB", (512, 512), (128,128,255))
img_roughness = Image.new("L", (512, 512), 160)
img_metallic =  Image.new("L", (512, 512), 0)

img = renderer.render(img_albedo,img_normal,img_roughness,img_metallic)
img.save(f"Results/Render/Image_{evalIndex}.png")  # Save the combined image#
