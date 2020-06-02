% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

clear all;clc
pram        = pram_init();

pram.Nx     = 128;
h2ax_mic    = DABBA_MU(pram);

%% train SEG
namestem = sprintf('Nx_%d_setB_ImgProc-rescale10k-softLbls',pram.Nx);
h2ax_mic.pram.miniBatchSize = 32;
h2ax_mic.pram.numEpochs     = 200;
h2ax_mic.pram.gammaMse      = 0;
%h2ax_mic.train_direct_imgprocessor;
tic
h2ax_mic.train_adversarial;
toc

%% train TRNS
namestem = sprintf('Nx_%d_shallow_setB_enc_dec',pram.Nx);
h2ax_mic.pram.numEpochs     = 4;
h2ax_mic.pram.gammaCyc      = 1e-3;
h2ax_mic.pram.miniBatchSize = 48;
h2ax_mic.pram.learnRateImgprocessor      = 0.0002;% 0.0002;
h2ax_mic.pram.learnRateDiscriminator     = 0.0001;% 0.0001;    
h2ax_mic.pram.learnRate_encoder          = 0.0002;% 0.0002;
h2ax_mic.pram.learnRate_decoder          = 0.0002;% 0.0002;

h2ax_mic.train_cycle;
save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic');

%% plots
savedir = ['./__results/exp_testTrainingCoditions/' namestem '_' date '/']
%plt_translation(h2ax_mic.imds_gt,h2ax_mic.imds_exp,h2ax_mic.encoder,h2ax_mic.decoder,savedir)
counts = plt_segmentation(h2ax_mic.imds_gt.subset(1:5),[],h2ax_mic.pxds_gt,[],[],h2ax_mic.imgprocessor,savedir);
counts = plt_segmentation(h2ax_mic.imds_gt            ,[],h2ax_mic.pxds_gt,[],[],h2ax_mic.imgprocessor,savedir);

counts = plt_segmentation(h2ax_mic.imds_exp.subset(1:3),[],[],[],[],h2ax_mic.imgprocessor,savedir);

% <formulate a loss plot next. Modify 'plt_trainingLosses.m'>

