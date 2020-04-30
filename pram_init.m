% 20200428 by Dushan N. Wadduwage
% edit this file to initiate parameters for the dabba_mu object

function pram = pram_init()
    
    pram.dir_dataroot   = '/home/harvard/Dropbox (Harvard University)/WorkingData/20200426_Bevin_Nuc/Nuclei Counter (20x, 40x)/20x Nuclei Counter/';
    pram.dir_imds_gt    = [pram.dir_dataroot 'Knime segmentation maps/Originals'];  
    pram.dir_imds_exp   = [pram.dir_dataroot 'Knime segmentation maps/Originals'];  
    pram.dir_pxds_gt    = [pram.dir_dataroot 'Knime segmentation maps/pxd'];  

    pram.ds_imread_channedls                = 1;
    pram.ds_imread_channedls_scale_method   = 'rescale10k';     % {'rescale10k','rescale1k','max','mean','none'} 
    pram.exp_type                           = 'h2ax_adv';       % {'h2ax_direct','h2ax_adv','h2ax'}

    pram.Nx                 = 128;
    pram.Nc                 = 1;
    pram.N_classes          = 2;                                % # segmented classes
    pram.classNames         = ["bg","fg"];
    pram.pxLblIds           = [0 1];
    
    pram.executionEnvironment   = 'auto';
    pram.gammaMse               = 0.01;

    pram.numEpochs                  = 1;      
    pram.miniBatchSize              = 48;
    pram.learnRateImgprocessor      = 0.0002;
    pram.learnRateDiscriminator     = 0.0001;    
    pram.gradientDecayFactor        = 0.5;
    pram.squaredGradientDecayFactor = 0.999;
    
    pram.trailingAvgDiscriminator   = [];
    pram.trailingAvgSqDiscriminator = [];    
    pram.trailingAvgImgprocessor    = [];
    pram.trailingAvgSqImgprocessor  = [];

end




