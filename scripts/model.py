import torch
from networks.base_model import BaseModel 
import networks.base_model as bs 
import networks.unet as unet
import networks.resnet as resnet
import networks.discriminator as discriminator
import networks.unet as unet
import torch.nn as nn
from torch.optim import lr_scheduler
from abc import ABC, abstractmethod
from collections import OrderedDict

class Model(ABC):
  
    def __init__(self, opt):
    
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 

        self.netG = unet.Unet(opt.input_channels,6,opt.nfilter)
        #self.netG = resnet.Resnet(opt.input_channels,6)

        self.netG= bs.init_net(self.netG, 'normal', 0.02, opt.gpu_ids)

        if self.isTrain: 

            self.netD =  discriminator.Discriminator(opt.input_channels+6, 64)
            self.netD = bs.init_net(self.netD, 'normal', 0.02, opt.gpu_ids)

            # define loss functions
            self.criterionGAN = bs.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            #self.criterionL1 = torch.nn.MSELoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            

            def lambda_rule2(epoch):
            
                epoch_s = opt.n_epochs * opt.decay_point 
                epoch_d = opt.n_epochs - epoch_s 

                lr_l = 1.0 - (max(0, epoch  - epoch_s) / float(epoch_d + 1))**2
                return lr_l

            
            self.schedulerG = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule2)
            self.schedulerD = lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule2)

    def lr_step(self):

        self.schedulerG.step()
        self.schedulerD.step()

    def set_input(self, input):
    
        self.Albedo = input['ALBEDO'].to(self.device)
        self.Real_PBR = input['PBR'].to(self.device)
        self.metadata_tensor= input['M'].to(self.device)

    def eval(self,input):
        
        self.netG.eval()
        self.netD.eval()
        
        with torch.no_grad():
            
            self.set_input(input)
            self.Fake_PBR = self.netG(self.Albedo,self.metadata_tensor) 

            fake_pair = torch.cat((self.Albedo, self.Fake_PBR), 1)

            pred_fake = self.netD(fake_pair)
            self.loss_D_fake = self.criterionGAN(pred_fake, False)

            self.eval_g_loss = self.criterionGAN(pred_fake, True)            
            self.eval_l1_loss = self.criterionL1(self.Fake_PBR, self.Real_PBR) 


    def optimize_parameters(self):

        # Set intro train mode
        self.netG.train()
        self.netD.train()
        
        # Produce Fake PBR image from input albedo
        self.Fake_PBR = self.netG(self.Albedo,self.metadata_tensor)  # G(A)

        # update D---------------------------------------
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()             # set D's gradients to zero
        
        # Fake; stop backprop to the generator by detaching fake_B
        real_pair = torch.cat((self.Albedo, self.Real_PBR), 1)
        fake_pair = torch.cat((self.Albedo, self.Fake_PBR), 1)  

        # Does it correctly classify as fake?
        pred_fake = self.netD(fake_pair.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Does it correctly classify as real?
        pred_real = self.netD(real_pair)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        self.optimizer_D.step()          # update D's weights

        # update G -------------------------------------
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero

        # First, G(A) should fake the discriminator
        pred_fake = self.netD(fake_pair)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.Fake_PBR, self.Real_PBR)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*self.opt.lambda_L1

        self.loss_G.backward()                  # calculate graidents for G
        self.optimizer_G.step()                 # update G's weights


    # Tools ---------------------------------------------------    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_visuals(self):

        visual_ret = OrderedDict()
        visual_ret["Albedo"] = self.Albedo
        visual_ret["Real_PBR"] = self.Real_PBR
        visual_ret["Fake_PBR"] = self.Fake_PBR

        return visual_ret
    
  