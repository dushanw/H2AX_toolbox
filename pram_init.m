% 20200428 by Dushan N. Wadduwage
% edit this file to initiate parameters for the dabba_mu object

function pram = pram_init()
    
    parm.dir_imds_gt    = '';    
    parm.dir_imds_exp   = '';
    parm.dir_pxds_gt    = '';
    pram.ds_imread_channedls                = 1;
    pram.ds_imread_channedls_scale_method   = 'rescale10k';     % {'rescale10k','rescale1k','max','mean','none'} 
    pram.exp_type                           = 'h2ax_simple';    % {'h2ax_simple','h2ax'}

    pram.Nx                 = 32;
    pram.Nc                 = 1;
    pram.N_classes          = 2;                                % # segmented classes
    pram.classNames         = ["bg","fg"];
    pram.pxLblIds           = [0 1];
    
end