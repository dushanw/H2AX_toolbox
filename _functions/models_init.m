
function models = models_init(pram)

    models.encoder = [];
    models.decoder = [];
    models.imgprocessor = [];
    models.discriminators = {[]};
    
    switch pram.exp_type
        case 'h2ax_direct'
            models.imgprocessor     = gennet_dncnnSegmenter(pram.Nx,pram.Nc,pram.N_classes);
        case 'h2ax_adv'
            models.encoder          = gendlnet_dummy           (pram.Nx,pram.Nc);                           % dummy network 
            models.decoder          = gendlnet_dummy           (pram.Nx,pram.Nc);
            models.imgprocessor     = gendlnet_dncnnSegmenter  (pram.Nx,pram.Nc,pram.N_classes);
            models.discriminators{1}= gendlnet_stdDiscriminator(pram.Nx,pram.Nc+pram.N_classes);
        case 'h2ax'
            models.encoder          = gendlnet_shallowDncnnImgTranslator(pram.Nx,pram.Nc);
            models.decoder          = gendlnet_shallowDncnnImgTranslator(pram.Nx,pram.Nc);
            models.imgprocessor     = gendlnet_dncnnSegmenter           (pram.Nx,pram.Nc,pram.N_classes);                      
            models.imgprocessor     = gendlnet_dncnnSegmenter           (pram.Nx,pram.Nc,pram.N_classes);
            models.discriminators{1}= gendlnet_stdDiscriminator         (pram.Nx,pram.Nc);                  % for gt & gt_hat
            models.discriminators{2}= gendlnet_stdDiscriminator         (pram.Nx,pram.Nc);                  % for measured & measured_hat
            models.discriminators{3}= gendlnet_stdDiscriminator         (pram.Nx,pram.Nc+pram.N_classes);   % for segmenter
    end
    
    
end