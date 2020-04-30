% 20200428 by Dushan N. Wadduwage

addpath(genpath('./_functions/'))
addpath(genpath('./_CLASSES/'))

parm        = pram_init();
h2ax_mic    = DABBA_MU(parm);

% h2ax_mic.train_direct_imgprocessor;
h2ax_mic.train_adversarial;

%% plot losses
plt_trainingLosses(h2ax_mic.tr_info)

