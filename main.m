% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

parm        = pram_init();
h2ax_mic    = DABBA_MU(parm);

% h2ax_mic.train_direct_imgprocessor;
h2ax_mic.train_adversarial;

% plot losses
figure;
subplot(1,2,1)
plot(h2ax_mic.tr_info.loss_imgprocessor);hold on
plot(h2ax_mic.tr_info.loss_discriminator);
legend('imgproc','desc')
subplot(1,2,2)
plot(h2ax_mic.tr_info.loss_discriminator_real);hold on
plot(h2ax_mic.tr_info.loss_discriminator_fake);
plot(h2ax_mic.tr_info.loss_imgprocessor_adv);
plot(h2ax_mic.tr_info.loss_imgprocessor_mse);
legend('desc real','desc fake','imgproc adv','imgproc mse')



