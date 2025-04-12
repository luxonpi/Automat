import clip
import torch
import pyiqa
import os, shutil
import util

clip_iqa = pyiqa.create_metric('clipiqa', device='cuda', as_loss=False)
fid_metric = pyiqa.create_metric('fid')

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def init(opt, val_dataset):

    # Validierung Datenzwischenspeichern
    if os.path.exists(os.path.join(opt.result_dir, "temp_real")):
            shutil.rmtree(os.path.join(opt.result_dir, "temp_real"))
    
    util.mkdirs(os.path.join(opt.result_dir, "temp_real"))

    for i, data in enumerate(val_dataset): 

        index= int(data["I"].item())
            
        image_normal_real=util.tensor2PIL(data['PBR'][:,0:3,:,:])
        image_roughness_real=util.tensor2PIL(data['PBR'][:,3,:,:].repeat(1,3, 1, 1) )
        image_height_real=util.tensor2PIL(data['PBR'][:,4,:,:].repeat(1,3, 1, 1) )
        image_metallic_real=util.tensor2PIL(data['PBR'][:,5,:,:].repeat(1,3, 1, 1) )
       
        image_normal_real.save(os.path.join(opt.result_dir, "temp_real", f'{index}_normal_real.png'))
        image_roughness_real.save(os.path.join(opt.result_dir, "temp_real", f'{index}_roughness_real.png'))
        image_height_real.save(os.path.join(opt.result_dir, "temp_real", f'{index}_height_real.png'))
        image_metallic_real.save(os.path.join(opt.result_dir, "temp_real", f'{index}_metallic_real.png'))
        

def clip_iqa_eval(image_set):

    clip_value_fake = clip_iqa(image_set["normal_fake"])
    clip_value_fake += clip_iqa(image_set["height_fake"])
    clip_value_fake += clip_iqa(image_set["roughness_fake"])
    clip_value_fake += clip_iqa(image_set["metallic_fake"])
    clip_value_fake = clip_value_fake/4

    clip_value_real = clip_iqa(image_set["normal_real"])
    clip_value_real += clip_iqa(image_set["height_real"])
    clip_value_real += clip_iqa(image_set["roughness_real"])
    clip_value_real += clip_iqa(image_set["metallic_real"])
    clip_value_real = clip_value_real/4
    
    #return dict
    return {"fake":clip_value_fake, "real":clip_value_real}


def Clip_Tag_Score_Dif(real,fake, tags):

    probs_real = Clip_Tag_Score(real, tags)
    probs_fake = Clip_Tag_Score(fake, tags)

    dif= probs_real - probs_fake

    # return the sum of the differences
    return dif.sum()


def Clip_Tag_Score(image, tags):

    image = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(tags).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        
        logits_per_image, logits_per_text = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs