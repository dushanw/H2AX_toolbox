% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

parm        = pram_init();
h2ax_mic    = DABBA_MU(parm);

% h2ax_mic.train_direct_imgprocessor;
% h2ax_mic.train_adversarial;

h2ax_mic.pram.gammaCyc = 0.001
h2ax_mic.train_cycle;

figure;
subplot(1,2,1);
plot(h2ax_mic.tr_info.loss_encoder);hold on;
plot(h2ax_mic.tr_info.loss_decoder	)
subplot(1,2,2);
plot(h2ax_mic.tr_info.loss_D_I);hold on;
plot(h2ax_mic.tr_info.loss_D_J)








%% plot losses
plt_trainingLosses(h2ax_mic.tr_info)

%% run test
h2ax_mic.imds_gt.reset;
h2ax_mic.pxds_gt.reset;

I_test      = h2ax_mic.imds_gt.read;
L_test_gt   = h2ax_mic.pxds_gt.read;
L_test_gt   = L_test_gt{1}=='fg';

dlI_test    = dlarray(I_test, 'SSCB');
dlL_test    = predict(h2ax_mic.imgprocessor,dlI_test);
L_test      = gather(dlL_test.extractdata);
L_test      = L_test(:,:,2)>.5;   

count       = plt_segmentation(I_test,L_test,L_test_gt,'./__results/pltseg_test_original')




