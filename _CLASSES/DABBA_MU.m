% 20200428 by Dushan N. Wadduwage
% Differentiable Microscope Class

classdef DABBA_MU < handle
  properties
    pram
    encoder
    decoder
    imgprocessor
    discriminators
    trData                   % GT-images + segmentation labels       
    expData                  % experimental images for I_Exp -> I_Exp_gtLike
    tr_info
  end 

  methods
    % constructor
    function obj = DABBA_MU(pram)
       obj.pram   = pram;
       obj.trData = load_trData(pram);    

       models               = models_init(pram);
       obj.encoder          = models.encoder;
       obj.decoder          = models.decoder;
       obj.imgprocessor     = models.imgprocessor;
       obj.discriminators   = models.discriminators;
    end

    function tr_segmenter_dirct(self)
       self.imgprocessor    = tr_segmenter_dirct(self.imgprocessor,self.trData,self.pram);
    end 

    function tr_segmenter_adv(self)
       [self.imgprocessor self.discriminators{3} self.tr_info] = ...
            tr_segmenter_adv(self.imgprocessor,self.discriminators{3},self.trData,self.pram);           
    end 

    function train_cycle(self)                
        [self.encoder self.decoder self.discriminators{1} self.discriminators{2} self.tr_info] = ...
            tr_translaters_cyc(self.encoder,...
                               self.decoder,...
                               self.discriminators{1},...
                               self.discriminators{2},...
                               self.trData,...                               
                               self.pram);
    end
  end

end








