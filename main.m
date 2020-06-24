% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_libs/'))
addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

clear all;clc
pram        = pram_init();
pram.Nx     = 256;
h2ax_mic    = DABBA_MU(pram);

%% train seg direct
% h2ax_mic.pram.miniBatchSize         = 256;
% h2ax_mic.pram.numEpochs             = 10;
% h2ax_mic.pram.learnRateDropFactor   = 0.1;
% h2ax_mic.pram.learnRateDropInterval = 10;
% h2ax_mic.pram.learnRateImgprocessor = 1;
% 
% h2ax_mic.tr_segmenter_dirct;  
% namestem = sprintf('Nx_%d_setB_ImgProc-rescale10k-directTrained-epochs_%d',pram.Nx,h2ax_mic.pram.numEpochs);
% save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic','-v7.3');
% 
% N_val = 10;
% C = semanticseg(h2ax_mic.trData.I_vl(:,:,:,1:N_val),h2ax_mic.imgprocessor,'OutputType','uint8');
% C = reshape(C,size(C,1),size(C,2),1,[]);
% I  = h2ax_mic.trData.I_vl(:,:,:,1:N_val);
% gt = h2ax_mic.trData.L_vl(:,:,1,1:N_val)+2*h2ax_mic.trData.L_vl(:,:,2,1:N_val);
% imagesc(imtile(cat(2,I,single(gt),single(C))));axis image
              
%% train SEG
% h2ax_mic.pram.miniBatchSize = 32;
% h2ax_mic.pram.numEpochs     = 50;
% h2ax_mic.pram.gammaMse      = 100;% 100 was working ok
% h2ax_mic.pram.learnRateImgprocessor      = 0.002;% 0.0002;
% h2ax_mic.pram.learnRateDiscriminator     = 0.001;% 0.0001;    
% 
% h2ax_mic.train_adversarial;
% namestem = sprintf('Nx_%d_setB_ImgProc-rescale10k-epochs_%d',pram.Nx,h2ax_mic.pram.numEpochs);
% save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic','-v7.3');
% 
% dir_dataroot  = '/home/harvard/Dropbox (Harvard University)/WorkingData/20200426_Bevin_Nuc/Nuclei Counter (20x, 40x)/20x Nuclei Counter/';
% dir_imds_exp  = [dir_dataroot '20.02.25 Ki67 24h set C']; % 
% imds_exp      = imageDatastore(dir_imds_exp,'ReadFcn',@(fname)ds_imread(fname,pram));
% savedir       = ['./__results/setC/' namestem '_' date '/'];
% counts        = plt_segmentation(imds_exp,[],[],[],[],h2ax_mic.imgprocessor,savedir);

 %% train TRNS
namestem = sprintf('Nx_%d__ENC-DEC_unet__H2AX_40x-wf-to-63x_max-zeroCenter',pram.Nx);
h2ax_mic.pram.numEpochs     = 6;
h2ax_mic.pram.miniBatchSize = 4;

% h2ax_mic.pram.learnRateImgprocessor  = 0.0002;% 0.0002;
h2ax_mic.pram.learnRateDiscriminator = 0.0001;% 0.0001;    
h2ax_mic.pram.learnRate_encoder      = 0.0002;% 0.0002;
h2ax_mic.pram.learnRate_decoder      = 0.0002;% 0.0002;
h2ax_mic.pram.learnRateDropFactor    = 0.1;
h2ax_mic.pram.learnRateDropInterval  = 2;
h2ax_mic.pram.gammaCyc               = 1;        
h2ax_mic.pram.gammaId                = 0.001;

h2ax_mic.train_cycle;
save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic');


