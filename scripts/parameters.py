class Parameters():


    dataset_dir = "/var/tmp/ge69fuv/full512"
    result_dir = "/var/tmp/ge69fuv/results"

    model_name_prefix = "base"

    # will be changed in automatic testing
    model_name = "base"
    input_channels = 3

    wandb_enabled = True
    wandb_name = "Automat"
    visual_eval_index = [5,7,11,18,19,24,28,33,35,45,47,48,51,54,62,70,79,82,83,86,89,95,103,104,112,119,127,128,148,160,161,162,164,165,167,168,172,176,178,189,194,196,204,206,210,212,218,219,226,244,249,253,259,265,266,276,282,294,302,303,324,326,330,347,355,366,377,382,387,407,408,414,415,426,431,437,458,463,470,476,477,497,538,550,551,555,556,582]

    gpu_ids = [0,1]           # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU

    # training parameters
    isTrain = True
    phase = "train"         # train, val, test, etc
    n_epochs = 300          
    decay_point = 0.3
    beta1 = 0.4             # momentum term of adam
    lr = 0.0006             # initial learning rate for adam
    gan_mode = "vanilla"    # the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
    lambda_L1 = 100
    maxTrainDataSize = 5000
    maxEvalDataSize = 1000
    batchSize = 8

    # model parameters
    nfilter = 64            # number of gen filters in the last conv layer
    n_layers_D = 3          # only used if netD==n_layers

    # Data augumentation
    random_rotate = False
    random_crop = False
    random_flip = False
    random_noise = False
    random_hue = False

    # Control
    height_factor_as_channel = False
    roughness_metallic_dist_as_channel = False
