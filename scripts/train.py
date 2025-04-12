import time
import torch
from parameters import Parameters
from dataset import Dataset,DatasetDataLoader
from model import Model
from logger import Logger
import os, shutil
import util
import json
import tqdm
import eval as eval
import math
import argparse

if __name__ == '__main__':

    #Remove Wandb tempory files if they exist -> creashes the script if multiple runs are started
    if os.path.exists("wandb"):
        shutil.rmtree("wandb")

    opt = Parameters()  

    #parser = argparse.ArgumentParser(description="Train the model with a specified learning rate.")
    #parser.add_argument("--beta", type=float, default=opt.beta1, help="Learning rate for training")
    #args = parser.parse_args()
    #opt.beta1 = args.beta

    opt.random_rotate = True
    opt.random_crop = True
    opt.random_hue = True
    opt.height_factor_as_channel = False
    opt.roughness_metallic_dist_as_channel = True

    opt.model_name = opt.model_name_prefix+ f"base_200_longrun"
        
    #if opt.height_factor_as_channel:
    #    opt.input_channels = 5

    if opt.roughness_metallic_dist_as_channel:
        opt.input_channels += 5

    # Datasets
    val_dataset = DatasetDataLoader(opt.dataset_dir+"/eval" ,opt,False,opt.maxEvalDataSize).load_data()
    train_dataset = DatasetDataLoader(opt.dataset_dir+"/train" ,opt,True,opt.maxTrainDataSize, opt.batchSize).load_data()

    eval.init(opt,val_dataset)

    # Metrics
    print('%d training images & %d validation images ' % (len(train_dataset), len(val_dataset)))
  
    train_dataset.dataset.opt = opt

    logger = Logger(opt) 
    model = Model(opt)     
    
    total_iterations = len(train_dataset)

    logger.NewRun(opt.model_name, opt)

    start_time= time.time()

    try:

        for epoch in range(0, opt.n_epochs):   

            # Create tqdm progress bar
            pbar = tqdm.tqdm(total=total_iterations, position=0, leave=False)

            # Training
            epoch_start_time = time.time() 
            train_loss = {'train_g':[], 'train_l1':[], 'train_d':[]}
            for i, data in enumerate(train_dataset): 
                
                model.set_input(data)         
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                
                #Append losses to history
                train_loss['train_g'].append(model.loss_G_GAN)
                train_loss['train_l1'].append(model.loss_G_L1)
                train_loss['train_d'].append(model.loss_D)

                pbar.update(opt.batchSize)
                
            epoch_time = time.time() - epoch_start_time
            model.lr_step()

            #close tqdm progress bar and remove it from outpur
            pbar.close()

            # Calulate average loss            
            train_loss_avg = {key: sum(value) / len(value) for key, value in train_loss.items()}

            # Validation
            # Evaluate the model        
            eval_loss = {'eval_g':[], 'eval_l1':[]}
            for i,data in enumerate(val_dataset): 
                
                model.eval(data)         

                #Append losses to history
                eval_loss['eval_g'].append(model.eval_g_loss)
                eval_loss['eval_l1'].append(model.eval_l1_loss)
                
            eval_loss_avg = {key: sum(value) / len(value) for key, value in eval_loss.items()}
            
            #Append train losses
            eval_loss_avg.update(train_loss_avg)
            eval_loss_avg["LearningRate"] = model.optimizer_G.param_groups[0]['lr']

            #logger.LogCurrentLosses(epoch, train_loss_avg,epoch_time)
            logger.PrintCurrentLosses(epoch, eval_loss_avg,epoch_time)
            logger.LogTrainLossesWandb(epoch, eval_loss_avg)
            
            if epoch % 10 == 0:              
                pass
                #print('Saving the model at the end of epoch %d' % (epoch))
                #model.save_networks('latest')
        
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        # Generate all Evaluation Images in Results/Temp
        # Log the chosens one to wand db
        # Final evaluation with FID and Clip and Images

        total_time = time.time() - start_time
        # in hours an min
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        # as string
        total_time_str = f"{hours}h {minutes}min"


        print("Final Evaluation "+ total_time_str)
        eval_loss = {'eval_g':[], 'eval_l1':[], 'clip-iqa-fake':[], 'clip-iqa-real':[]}

        # clear the folling paths os.path.join(opt.result_dir, "temp_real",
        if os.path.exists(os.path.join(opt.result_dir, "temp_fake")):
            shutil.rmtree(os.path.join(opt.result_dir, "temp_fake"))
        
        util.mkdirs(os.path.join(opt.result_dir, "temp_fake"))
        
        pbar2 = tqdm.tqdm(total=len(val_dataset), position=0, leave=False)

        for i, data in enumerate(val_dataset): 
            
            index= int(data["I"].item())

            model.eval(data)    

            image_set = util.create_image_set(model.get_current_visuals())
            logger.SaveTmpImages(image_set,opt, index)

            eval_loss['eval_g'].append(model.eval_g_loss)
            eval_loss['eval_l1'].append(model.eval_l1_loss)

            clipiq_values = eval.clip_iqa_eval(image_set)
            eval_loss['clip-iqa-fake'].append(clipiq_values["fake"])
            eval_loss['clip-iqa-real'].append(clipiq_values["real"])

            #renders=logger.RenderImageSet(image_set,opt, index)
            meta_path = os.path.join(opt.dataset_dir, "eval", f'{index}_meta.json')
            metadata = {}
            tags = []
            with open(meta_path, 'r') as file:
                metadata = json.load(file)
                tags = metadata["tags"]

            if index in opt.visual_eval_index:
                logger.LogVisuals(model.get_current_visuals(),index,metadata)

            pbar2.update(1)

        pbar2.close()
        
        print("FID eval")

        fid_score = eval.fid_metric(os.path.join(opt.result_dir, "temp_fake"), os.path.join(opt.result_dir, "temp_real"))

        eval_loss_avg = {key: sum(value) / len(value) for key, value in eval_loss.items()}
        eval_loss_avg["FID"] = fid_score

        logger.LogFinalEvaluationMetrics(opt.model_name, opt, eval_loss_avg,total_time_str)
        logger.FinishRun()

        # Save the model
        print('Saving the model at the end of training')
        save_path = os.path.join('/var/tmp/ge69fuv/', 'gnet.pth')
        torch.save(model.netG.cpu().state_dict(), save_path)

    except KeyboardInterrupt:
        
        print("\nTraining interrupted! Saving checkpoint...")
        save_path = os.path.join('/var/tmp/ge69fuv/', 'gnet.pth')
        torch.save(model.netG.cpu().state_dict(), save_path)
