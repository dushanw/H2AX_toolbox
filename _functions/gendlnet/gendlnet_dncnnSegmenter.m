function dlnet = gendlnet_dncnnSegmenter(Nx,Nc, N_classes)

    sizeIn = [Nx Nx Nc];
    
    net0_dncnn = denoisingNetwork('dncnn');
    lgraph = layerGraph(net0_dncnn.Layers);
    
    lgraph = replaceLayer(lgraph,'InputLayer',imageInputLayer(sizeIn,'Name','InputLayer','Normalization','none'));
    lgraph = replaceLayer(lgraph,'Conv20',convolution2dLayer(1,N_classes,'Name','Conv20'));
    lgraph = removeLayers(lgraph,'FinalRegressionLayer');
    lgraph = addLayers(lgraph,softmaxLayer('Name','Softmax'));
    
    % convert relu to leakyRelu
    for i=1:19
        lgraph = replaceLayer(lgraph,sprintf('ReLU%d',i),leakyReluLayer('Name',sprintf('LeakyRelu%d',i)));        
    end
    
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_2_5'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_5_10'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_10_15'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_15_19'));
    
    lgraph = disconnectLayers(lgraph,'BNorm5','LeakyRelu5');
    lgraph = disconnectLayers(lgraph,'BNorm10','LeakyRelu10');
    lgraph = disconnectLayers(lgraph,'BNorm15','LeakyRelu15');
    lgraph = disconnectLayers(lgraph,'BNorm19','LeakyRelu19');
    
    lgraph = connectLayers(lgraph,'BNorm5','add_2_5/in1');
    lgraph = connectLayers(lgraph,'LeakyRelu2','add_2_5/in2');
    lgraph = connectLayers(lgraph,'add_2_5','LeakyRelu5');
    
    lgraph = connectLayers(lgraph,'BNorm10','add_5_10/in1');
    lgraph = connectLayers(lgraph,'LeakyRelu5','add_5_10/in2');
    lgraph = connectLayers(lgraph,'add_5_10','LeakyRelu10');
    
    lgraph = connectLayers(lgraph,'BNorm15','add_10_15/in1');
    lgraph = connectLayers(lgraph,'LeakyRelu10','add_10_15/in2');
    lgraph = connectLayers(lgraph,'add_10_15','LeakyRelu15');
    
    lgraph = connectLayers(lgraph,'BNorm19','add_15_19/in1');
    lgraph = connectLayers(lgraph,'LeakyRelu15','add_15_19/in2');
    lgraph = connectLayers(lgraph,'add_15_19','LeakyRelu19');
    
    lgraph = connectLayers(lgraph,'Conv20','Softmax');
    
%     lgraph = addLayers(lgraph,dicePixelClassificationLayer('Name','final_pxClasLayer'));
%     lgraph = connectLayers(lgraph,'Softmax','final_pxClasLayer');
    
    dlnet  = dlnetwork(lgraph);    
end