% 20200428 by Dushan N. Wadduwage
% Differentiable Microscope Class

classdef DABBA_MU < handle
   properties
       pram
       encoder
       decoder
       imgprocessor
       discriminators
       imds_gt                  % ground truth images used for training       
       imds_exp                 % dabba_mu measured images
       pxds_gt                  % ground truth annotations
   end 
   
   methods
       % constructor
       function obj = DABBA_MU(pram)
           obj.pram     = pram;

           obj.imds_gt  = imageDatastore     (parm.dir_imds_gt      ,'ReadFcn',@(fname)ds_imread(fname,pram));           
           obj.imds_exp = imageDatastore     (parm.dir_imds_exp     ,'ReadFcn',@(fname)ds_imread(fname,pram));
           obj.pxds_gt  = pixelLabelDatastore(parm.dir_pxds_gt      ,parm.classNames,parm.pixelLabelIDs);
           
           models       = models_init(pram)
           obj.encoder          = models.encoder;
           obj.decoder          = models.decoder;
           obj.imgprocessor     = models.imgprocessor;
           obj.discriminators   = models.discriminators;
       end

       function train_direct_imgprocessor(self)
           self.imgprocessor    = tr_segmenter(self.imgprocessor,self.imds_gt,self.pxds_gt,self.pram)
       end 

       function train_adversarial(self)
           % 
       end 


   end
   
end







