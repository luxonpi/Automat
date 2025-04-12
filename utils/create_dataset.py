import os
from PIL import Image
from tqdm import tqdm
import json
import random

# Paths for input and output directories
input_dir = 'Datasets/MatSynth/train'  # Directory containing the image pairs
output_dir = 'Datasets/Automat/full512'  # Directory to save the combined images


train_indices = list(range(5699))  # Includes 0 through 6000

# Calculate 10% and 20% of the total number of indices
ten_percent_count = int(len(train_indices) * 0.05)
twenty_percent_count = int(len(train_indices) * 0.15)

test_indices = random.sample(train_indices, ten_percent_count)
train_indices = list(set(train_indices) - set(test_indices))

eval_indices = random.sample(train_indices, twenty_percent_count)
train_indices = list(set(train_indices) - set(eval_indices))

print(len(train_indices))
print(len(test_indices))
print(len(eval_indices))

print(len(eval_indices+test_indices+train_indices))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

#create train and eval
os.makedirs(output_dir+"/train", exist_ok=True)
os.makedirs(output_dir+"/eval", exist_ok=True)
os.makedirs(output_dir+"/test", exist_ok=True)

# Get all files in the input directory
files = sorted(os.listdir(input_dir))


progress_bar = tqdm(total=5699, desc="Image Set", leave=False)

def saveIndex(index, folder, counter):

    albedo_path = os.path.join(input_dir, f"{index}_albedo.png")  # Albedo image path
    rougness_path = os.path.join(input_dir, f"{index}_roughness.png")  # Normal image path
    normal_path = os.path.join(input_dir, f"{index}_normal.png")  # Normal image path
    metallic_path = os.path.join(input_dir, f"{index}_metallic.png")  # Normal image path
    height_path = os.path.join(input_dir, f"{index}_height.png")  # Normal image path
    meta = os.path.join(input_dir, f"{index}_metadata.json")  # Normal image path

    # Open both images
    albedo_image = Image.open(albedo_path).resize((512, 512))
    roughness_image = Image.open(rougness_path).resize((512, 512))
    normal_image = Image.open(normal_path).resize((512, 512))
    metallic_image = Image.open(metallic_path).resize((512, 512))
    height_image = Image.open(height_path).resize((512, 512))

    with open(meta, 'r') as file:
        data = json.load(file)
    
    save_folder= os.path.join(output_dir, folder)

    albedo_image.save(os.path.join(save_folder, f"{counter}_albedo.png"))
    roughness_image.save(os.path.join(save_folder, f"{counter}_roughness.png"))
    normal_image.save(os.path.join(save_folder, f"{counter}_normal.png"))
    metallic_image.save(os.path.join(save_folder, f"{counter}_metallic.png"))
    height_image.save(os.path.join(save_folder, f"{counter}_height.png"))

    string_path = os.path.join(save_folder, f"{counter}_meta.json")  
    with open(string_path, 'w') as file:
        file.write(json.dumps(data, indent=4))
            

#Train
for counter, index in enumerate(train_indices):

    saveIndex(index,"train",counter)
    progress_bar.update(1) 

#Test
for counter, index in enumerate(test_indices):

    saveIndex(index,"test",counter)
    progress_bar.update(1) 

#Eval
for counter, index in enumerate(eval_indices):

    saveIndex(index,"eval",counter)
    progress_bar.update(1) 




  

            


