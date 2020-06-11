% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

clear all;clc
pram        = pram_init();
pram.Nx     = 64;
h2ax_mic    = DABBA_MU(pram);

%% train SEG
namestem = sprintf('Nx_%d_setB_ImgProc-rescale10k-softLbls',pram.Nx);
h2ax_mic.pram.miniBatchSize = 192;
h2ax_mic.pram.numEpochs     = 10;
h2ax_mic.pram.gammaMse      = 100;
h2ax_mic.pram.learnRateImgprocessor      = 0.002;% 0.0002;
h2ax_mic.pram.learnRateDiscriminator     = 0.001;% 0.0001;    

h2ax_mic.train_adversarial;
save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic');

dir_dataroot  = '/home/harvard/Dropbox (Harvard University)/WorkingData/20200426_Bevin_Nuc/Nuclei Counter (20x, 40x)/20x Nuclei Counter/';
dir_imds_exp  = [dir_dataroot '20.02.24 Ki67 24h set B']; %  '20.02.21 Ki67 24h set A', '20.02.24 Ki67 24h set B' 
imds_exp      = imageDatastore(dir_imds_exp,'ReadFcn',@(fname)ds_imread(fname,pram));
savedir       = ['./__results/setB/' namestem '_' date '/'];
counts        = plt_segmentation(imds_exp,[],[],[],[],h2ax_mic.imgprocessor,savedir);



 %% train TRNS
% namestem = sprintf('Nx_%d_shallow1_setB_enc_dec-softlbls',pram.Nx);
% h2ax_mic.pram.numEpochs     = 5;
% h2ax_mic.pram.gammaCyc      = 10;
% h2ax_mic.pram.miniBatchSize = 64;
% h2ax_mic.pram.learnRateImgprocessor      = 0.0002;% 0.0002;
% h2ax_mic.pram.learnRateDiscriminator     = 0.0001;% 0.0001;    
% h2ax_mic.pram.learnRate_encoder          = 0.0002;% 0.0002;
% h2ax_mic.pram.learnRate_decoder          = 0.0002;% 0.0002;
% 
% h2ax_mic.train_cycle;
% % save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic');


