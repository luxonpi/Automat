This is a GAN architecture based on Pix2Pix whicht is trained to produce whole PBR Texture Sets from an input albedo.
Additional controll over the generation process is provided by adding the desired value distributions of the target textures .

Install Python Dependencies:
pip install -r requirements.txt

Download Dataset: 
https://huggingface.co/datasets/gvecchio/MatSynth  
put the folder called "Dataset" in the root project folder. It should contain "train" and "eval" folders with the images called 0_albedo.png 0_height.png etc.
Scale them down if needed.  

train gan: sh run.sh

Test trained gan (height mean, height variance, roughness mean, roughness variance, metallic mean, metallic variance ):  
python test.py <image.jpg> <[hm,hv,rm,rv,mm,mv]>")


This work was made in context of a Master Thesis at the LDV Chair at the Technical University of Munich.  
