import numpy as np
import os
import sys
import ntpath
import time
import util
from PIL import Image, ImageDraw, ImageFont
from subprocess import Popen, PIPE
import torch
import renderer as renderer
from datetime import datetime

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

# Settings
font_size = 12 
font_path = "resources/HelveticaNeueMedium.otf" 
font = ImageFont.truetype(font_path, font_size)
background_color = "white"
font_color = "black"

class Logger():
 
    def __init__(self, parameters):
      
        self.paramters = parameters  
        self.use_wandb = parameters.wandb_enabled
        self.wandb_project_name = parameters.wandb_name


    
    def NewRun(self, name, opt):

        if self.use_wandb:
            self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.model_name, config= self.paramters) if not wandb.run else wandb.run

        self.current_epoch = 0    
        self.img_dir = os.path.join(opt.result_dir, opt.model_name, 'images')
        util.mkdirs([self.img_dir])

    
    def AddParametersToImage(self, image, losses, parameters):

        # turn parameter dictionary into a string
        text = ""
        for key, value in losses.items():
            text += f"{key}: {value:.4f} | "

        #add linebreak
        text += "\n"

        keyToShow = ["epoch","name","category"]
        for item in keyToShow:
            if item in parameters:
                key = item
                value = parameters[item]

                text += f"{key}: {value} | "

        text += "\n"
        keyToShow = ["height_factor","height_mean","tags"]
        for item in keyToShow:
            if item in parameters:
                key = item
                value = parameters[item]

                #if value is type float tensor convert to 4 digit string
                if isinstance(value, torch.Tensor):
                    titem = value.item()
                    value = f"{titem:.4f}"

                text += f"{key}: {value} | "

        labels = ["ALBEDO", "NORMAL", "ROUGHNESS", "HEIGHT", "PBR"]

        # Calculate text size
        text_bbox = font.getbbox(text)  # Returns (left, top, right, bottom)

        text_width = text_bbox[2] - text_bbox[0]  # right - left
        text_height = text_bbox[3] - text_bbox[1]  # bottom - top

        bar_height = 50 # Add padding above and below the text
        image_width, image_height = image.size

        # Create a new image with extra space for the black bar
        new_image = Image.new("RGB", (image_width, image_height + bar_height), background_color)
        new_image.paste(image, (0, 0))  # Paste the original image onto the new one

        # Draw text on the black bar
        draw = ImageDraw.Draw(new_image)

        x_position = 20  # Align to the left with 10px padding
        y_position = image_height+50 - text_height-35  # Center vertically in the black bar
     
        # Add white text with a black drop shadow
        draw.text((x_position, y_position), text, font=font, fill=font_color)          # Main text

        stepsize = image_width // len(labels)  # Calculate the width of each label
        for i, label in enumerate(labels):
            text_bbox = font.getbbox(label)  # Returns (left, top, right, bottom)
            text_width = text_bbox[2] - text_bbox[0]  # right - left

            x_position = i * stepsize + stepsize // 2 - text_width // 2
            y_position = image_height/2 -5
            draw.text((x_position, y_position), label, font=font, fill=font_color)

        return new_image
    
    def SaveTmpImages(self, image_set, opt, index):

        image_set["metallic_fake"].save(os.path.join(opt.result_dir, "temp_fake", f'{index}_metallic.png'))
        image_set["roughness_fake"].save(os.path.join(opt.result_dir, "temp_fake", f'{index}_roughness.png'))
        image_set["height_fake"].save(os.path.join(opt.result_dir, "temp_fake", f'{index}_height.png'))
        image_set["normal_fake"].save(os.path.join(opt.result_dir, "temp_fake", f'{index}_normal.png'))
 
    def RenderImageSet(self, image_set, opt, index):
 
        rendered_real = renderer.render(image_set["albedo"], image_set["normal_real"],image_set["roughness_real"],image_set["metallic_real"]) 
        rendered_real.save(os.path.join(opt.result_dir, "rendered_real", f'{index}_rendered.png'))

        rendered_fake = renderer.render(image_set["albedo"], image_set["normal_fake"],image_set["roughness_fake"],image_set["metallic_fake"]) 
        rendered_fake.save(os.path.join(opt.result_dir, "rendered_fake", f'{index}_rendered.png'))

        # Create black image of the same size
        black_image = Image.new("RGB", (256, 256), "gray")
        rendered_real_b = renderer.render(black_image, image_set["normal_real"],image_set["roughness_real"],image_set["metallic_real"]) 
        rendered_real_b.save(os.path.join(opt.result_dir, "rendered_real", f'{index}_rendered_b.png'))

        rendered_fake_b = renderer.render(black_image, image_set["normal_fake"],image_set["roughness_fake"],image_set["metallic_fake"]) 
        rendered_fake_b.save(os.path.join(opt.result_dir, "rendered_fake", f'{index}_rendered_b.png'))

        #return dict
        return {"real": rendered_real, "fake": rendered_fake, "real_b": rendered_real_b, "fake_b": rendered_fake_b}
       
    def RenderPbr(self, albedo, pbr, save_name):

        image_normal=util.tensor2im(pbr[:,0:3,:,:])
        image_roughness=util.tensor2im(pbr[:,3,:,:].repeat(1,3, 1, 1) )
        image_height=util.tensor2im(pbr[:,4,:,:].repeat(1,3, 1, 1) )
        image_metallic=util.tensor2im(pbr[:,5,:,:].repeat(1,3, 1, 1) )
        image_albedo=util.tensor2im(albedo[:,0:3,:,:])

        #img_albedo, img_normal, img_roughness, img_metallic
        rendered_r = renderer.render(Image.fromarray(image_albedo), Image.fromarray(image_normal),Image.fromarray(image_roughness),Image.fromarray(image_metallic)) 
    
        rendered_r.save(save_name)  # Save the combined image#


    def LogVisuals(self, visuals, epoch, parameters={}):

        image_normal_real=visuals['Real_PBR'][:,0:3,:,:]
        image_normal_fake=visuals['Fake_PBR'][:,0:3,:,:]

        image_roughness_real=visuals['Real_PBR'][:,3,:,:].repeat(1,3, 1, 1) 
        image_roughness_fake=visuals['Fake_PBR'][:,3,:,:].repeat(1,3, 1, 1) 

        image_height_real=visuals['Real_PBR'][:,4,:,:].repeat(1,3, 1, 1) 
        image_height_fake=visuals['Fake_PBR'][:,4,:,:].repeat(1,3, 1, 1) 
      
        image_metallic_real=visuals['Real_PBR'][:,5,:,:].repeat(1,3, 1, 1) 
        image_metallic_fake=visuals['Fake_PBR'][:,5,:,:].repeat(1,3, 1, 1) 

        image_albedo_real=visuals['Albedo'][:,0:3,:,:]

        images_top=[image_albedo_real,image_normal_real,image_roughness_real,image_height_real]
        images_bottom=[image_albedo_real,image_normal_fake,image_roughness_fake, image_height_fake]

        images_top = [util.tensor2im(image) for image in images_top]  # Convert all tensors to numpy arrays
        images_bottom = [util.tensor2im(image) for image in images_bottom]  # Convert all tensors to numpy arrays

        rendered_r = renderer.render(Image.fromarray(images_top[0]), Image.fromarray(images_top[1]),Image.fromarray(images_top[2]),Image.fromarray(util.tensor2im(image_metallic_real))) 
        rendered_b = renderer.render(Image.fromarray(images_bottom[0]), Image.fromarray(images_bottom[1]),Image.fromarray(images_bottom[2]),Image.fromarray(util.tensor2im(image_metallic_fake))) 

        rendered_r = rendered_r.convert("RGB").resize((256, 256))
        rendered_b = rendered_b.convert("RGB").resize((256, 256))
        images_top.append(np.array(rendered_r))
        images_bottom.append(np.array(rendered_b))

        combined_image_u = Image.fromarray(np.hstack(images_top))  # Stack images horizontally
        combined_image_b = Image.fromarray(np.hstack(images_bottom))  # Stack images horizontally

        # Create a black spacer image
        spacer = Image.new("RGB", (combined_image_u.width, 20), background_color)

        # Stack images with spacer in between
        combined_image = Image.fromarray(np.vstack([combined_image_u, np.array(spacer), combined_image_b]))

        img_path = os.path.join(self.img_dir, f'epoch{epoch:03d}_combined.png')

        # add epoch to begginging of parameters
        parameters = {"epoch": epoch, **parameters}

        combined_image=self.AddParametersToImage(combined_image, {}, parameters)

        combined_image.save(img_path)  # Save the combined image#

        if self.use_wandb:
            log_data = {
                "Eval": wandb.Image(combined_image),  # Include combined image
            }
            self.wandb_run.log(log_data)  # Log everything at once


    def LogTrainLossesWandb(self, epoch, losses):

        if self.use_wandb:
            log_data = {
                **losses,  # Include losses
            }
            self.wandb_run.log(log_data)  # Log everything at once

    # Save the current losses to a text file -----------------------------------
    def FinishRun(self):
        if self.use_wandb:
            wandb.finish()

    def PrintCurrentLosses(self, epoch, losses,epoch_time):
    
        message = '(epoch: %d, time: %d) ' % (epoch, epoch_time)
        for k, v in losses.items():
            message += '%s: %.5f ' % (k, v)

        print(message)

    def LogFinalEvaluationMetrics(self,name, opt, losses, time):
        
        evallog = os.path.join(opt.result_dir, 'evaluation.txt')
        now = datetime.now().strftime('%m/%d %H:%M')

        # Time in h and min
        
        message = name+time
        for k, v in losses.items():
            message += '%s: %.5f ,' % (k, v)

        with open(evallog, "a") as log_file:
            log_file.write(message+"\n")
        
