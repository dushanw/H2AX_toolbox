% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

pram        = pram_init();

pram.Nx     = 128;
h2ax_mic    = DABBA_MU(pram);

%% train SEG
% h2ax_mic.train_direct_imgprocessor;
h2ax_mic.train_adversarial;

%% train TRNS
namestem = sprintf('Nx_%d_deep_enc_dec',pram.Nx);
h2ax_mic.pram.numEpochs     = 2;
h2ax_mic.pram.gammaCyc      = 0.1;
h2ax_mic.pram.miniBatchSize = 12;         
h2ax_mic.train_cycle;
save(['./_trainedNetworks/' sprintf('%s_%s.mat',date,namestem)],'h2ax_mic');

%% plots
savedir = ['./__results/exp_throughEncDec/' namestem '/'];
plt_translation(h2ax_mic.imds_gt,h2ax_mic.imds_exp,h2ax_mic.encoder,h2ax_mic.decoder,savedir)
counts = plt_segmentation(h2ax_mic.imds_gt.subset(1:2),[],h2ax_mic.pxds_gt,[],[],h2ax_mic.imgprocessor,savedir)
% <formulate a loss plot next. Modify 'plt_trainingLosses.m'>

