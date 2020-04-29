
function dlnetDiscriminator = gendlnet_stdDiscriminator(Nx,Nc)
    
    filterSize              = [4 4];
    N_middleLayers          = log2(Nx)-2;        
    N_filters               = 64;
    scale                   = 0.2;
    
    layersDiscriminator     = [
        imageInputLayer([Nx Nx Nc],'Normalization','none','Name','in')
        convolution2dLayer(filterSize,N_filters,'Stride',2,'Padding',1,'Name','conv1')
        leakyReluLayer(scale,'Name','lrelu1')
        ];
    
    for i=2:N_middleLayers    
        N_filters           = 2*N_filters;
        layersDiscriminator = [
            layersDiscriminator
            convolution2dLayer(filterSize,N_filters,'Stride',2,'Padding',1,'Name',sprintf('conv%d',i))
            batchNormalizationLayer('Name',sprintf('bn%d',i))
            leakyReluLayer(scale,'Name',sprintf('lrelu%d',i))
            ];
    end
    
    layersDiscriminator = [
            layersDiscriminator
            convolution2dLayer(filterSize,1,'Name',sprintf('conv%d',i+1))
            ];
        
    lgraphDiscriminator = layerGraph(layersDiscriminator);
    dlnetDiscriminator = dlnetwork(lgraphDiscriminator);
end
            
        
            
            
  