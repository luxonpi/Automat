import os
from os.path import abspath, isdir, join, basename
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
import random
import numpy as np
import torch
import json
from random import randrange
from PIL import ImageFile
import re
from random import uniform

IMG_EXTENSIONS = ['.png', '.PNG']

# ------------------- Dataset loader ------------------- #
class DatasetDataLoader():

    def __init__(self, folder, opt, isTrain, maxSize, batch_size=1):
     
        self.batch_size = batch_size
        self.dataset = Dataset(folder,opt,isTrain,maxSize)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=isTrain, num_workers=4, persistent_workers=True)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= len(self.dataset):
                break
            yield data

# ------------------- Dataset class ------------------- #

class Dataset():

    def __init__(self, folder, opt, isTrain,maxSize):
        self.opt = opt
        self.isTrain = isTrain
        self.AB_paths = sorted(self.make_dataset(folder,maxSize))  # get image paths
    
    def is_image_file(self, filename):
        return any(filename.endswith("_albedo" + extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, dir, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

  
    def __len__(self):
        return len(self.AB_paths)
    
 

    def apply_tranform(self, image,flipX,flipY,rotateAngle,crop_size):
        
        imageMode= image.mode
        
        # -------- Train Only ------------ #
        if self.isTrain:
            
            if flipX and self.opt.random_flip:
                image=transforms.functional.hflip(image)
            if flipY and self.opt.random_flip:
                image=transforms.functional.vflip(image)

            if self.opt.random_rotate:
                avg_image = image.resize((1, 1), resample=Image.Resampling.BILINEAR)
                avg_color = avg_image.getpixel((0, 0))
                image=image.rotate(rotateAngle, Image.BILINEAR, expand = 0, fillcolor = avg_color)

            if self.opt.random_crop:
                image=transforms.functional.center_crop(image, (crop_size, crop_size))
        
        image= transforms.functional.resize(image, (256, 256))

        # -------- Train & Test ------------ #
        image= transforms.functional.to_tensor(image)

        if imageMode == 'RGB':
            image=  transforms.functional.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        elif imageMode == 'L':
            image=  transforms.functional.normalize(image, mean=(0.5,), std=(0.5,))

        return image

    def __getitem__(self, index):
        
        # read a image given a random integer index
        albedo_path = self.AB_paths[index]

        match = re.search(r'/(\d+)_albedo', albedo_path)

        image_index=index
        if match:
            image_index =int( match.group(1))
       
        height_path = albedo_path.replace("albedo.png", "height.png")
        metallic_path = albedo_path.replace("albedo.png", "metallic.png")
        roughness_path = albedo_path.replace("albedo.png", "roughness.png")
        normal_path = albedo_path.replace("albedo.png", "normal.png")
        meta_path = albedo_path.replace("albedo.png", "meta.json")

        metadata = {}
        with open(meta_path, 'r') as file:
            metadata = json.load(file)

        #get entry "height_factor" from metadata as float
        height_factor = float(metadata["height_factor"])
        height_mean = float(metadata["height_mean"])

        #create a 512x512 tensor with the height factor
        height_factor_t = torch.full((1, 256, 256), height_factor*100)
        height_mean_t = torch.full((1, 256, 256), height_mean*100)

        crop_size = 256+randrange(256)

        flipX = random.choice([0,1])
        flipY = random.choice([0,1])

        rotateAngle = randrange(20)-10

        ALEBDO = Image.open(albedo_path).convert('RGB')
        normal = Image.open(normal_path).convert('RGB')
        roughness = Image.open(roughness_path).convert('L')
        height = Image.open(height_path).convert('L')
        metallic = Image.open(metallic_path).convert('L')

        if self.opt.random_hue & self.isTrain:
        
            # Define a transform for color jitter
            color_jitter = transforms.ColorJitter(
                brightness=0.4,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )

            # Apply the transform
            ALEBDO = color_jitter(ALEBDO)

        if self.opt.random_noise:
            gaussian_noise = torch.normal(mean=0.0, std=0.1, size=ALEBDO.shape)
            ALEBDO = (ALEBDO + gaussian_noise)
     
        ALEBDO = self.apply_tranform(ALEBDO,flipX,flipY,rotateAngle,crop_size)
        normal = self.apply_tranform(normal,flipX,flipY,rotateAngle,crop_size)
        roughness= self.apply_tranform(roughness,flipX,flipY,rotateAngle,crop_size)
        height=self.apply_tranform(height,flipX,flipY,rotateAngle,crop_size)
        metallic=self.apply_tranform(metallic,flipX,flipY,rotateAngle,crop_size)

        #if self.opt.height_factor_as_channel:
        #    ALEBDO = torch.cat((ALEBDO, height_factor_t,height_mean_t), 0)

        if self.opt.roughness_metallic_dist_as_channel:
            # Calculate mean and variance for metallic and roughness tensors
            metallic_mean = torch.mean(metallic)
            metallic_variance = torch.var(metallic)
            roughness_mean = torch.mean(roughness)
            roughness_variance = torch.var(roughness)
            height_mean = torch.mean(height)
            height_variance = torch.var(height)

            # Create tensors for the calculated statistics
            metallic_mean_t = torch.full((1, metallic.shape[1], metallic.shape[2]), metallic_mean.item())
            metallic_variance_t = torch.full((1, metallic.shape[1], metallic.shape[2]), metallic_variance.item())
           
            roughness_mean_t = torch.full((1, roughness.shape[1], roughness.shape[2]), roughness_mean.item())
            roughness_variance_t = torch.full((1, roughness.shape[1], roughness.shape[2]), roughness_variance.item())

            #height_mean_t = torch.full((1, height.shape[1], height.shape[2]), height_mean.item())
            height_variance_t = torch.full((1, height.shape[1], height.shape[2]), height_variance.item())

            ALEBDO = torch.cat((ALEBDO, metallic_mean_t, metallic_variance_t, roughness_mean_t, roughness_variance_t,height_variance_t), 0)


        PBR = torch.cat((normal, roughness,height,metallic), 0)


        metallic_mean = torch.mean(metallic)
        metallic_variance = torch.var(metallic)
        roughness_mean = torch.mean(roughness)
        roughness_variance = torch.var(roughness)
        height_mean = torch.mean(height)
        height_variance = torch.var(height)

        # combine all mean an variance into one tensor
        metadata_tensor = torch.tensor([metallic_mean, metallic_variance, roughness_mean, roughness_variance, height_mean, height_variance])    


        return {'ALBEDO': ALEBDO, 'PBR': PBR, 'I': image_index, 'M': metadata_tensor}