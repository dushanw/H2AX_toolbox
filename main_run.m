% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_libs/'))
addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

clear all;clc

netname   = '14-Jun-2020_Nx_128_setB_ImgProc-rescale10k-30epochs.mat';% copy the name of the saved file
namestem  = extractBetween(netname,'2020_','.mat');
namestem  = namestem{1};
expname   = '20.02.28 Ki67 24h set D';

load(['./_trainedNetworks/' netname])
pram                      = h2ax_mic.pram;
pram.ds_imread_channedls  = 2;

dir_dataroot  = '/home/harvard/Dropbox (Harvard University)/WorkingData/20200426_Bevin_Nuc/Nuclei Counter (20x, 40x)/20x Nuclei Counter/';
dir_imds_exp  = [dir_dataroot expname]; % 
imds_exp      = imageDatastore(dir_imds_exp,'ReadFcn',@(fname)ds_imread(fname,pram));
savedir       = ['./__results/' expname '/' namestem '_' date '/'];
counts        = plt_segmentation(imds_exp,[],[],[],[],h2ax_mic.imgprocessor,savedir);

 %% train TRNS
% namestem = sprintf('Nx_%d__ENC-DEC_dncnn__DAPI_40x-wf-to-63x',pram.Nx);
% h2ax_mic.pram.numEpochs     = 10;
% h2ax_mic.pram.gammaCyc      = 10;
% pram.gammaId                = 0;
% h2ax_mic.pram.miniBatchSize = 8;
% 
% pram.learnRateImgprocessor  = 0.002;% 0.0002;
% pram.learnRateDiscriminator = 0.001;% 0.0001;    
% pram.learnRate_encoder      = 0.002;% 0.0002;
% pram.learnRate_decoder      = 0.002;% 0.0002;
% pram.learnRateDropFactor    = 0.5;
% pram.learnRateDropInterval  = 1;% hard-coded-to-one now
% h2ax_mic.train_cycle;
% save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic');


