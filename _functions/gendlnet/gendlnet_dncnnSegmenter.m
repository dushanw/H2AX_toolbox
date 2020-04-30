function dlnet = gendlnet_dncnnSegmenter(Nx,Nc, N_classes)

    sizeIn = [Nx Nx Nc];
    
    net0_dncnn = denoisingNetwork('dncnn');
    lgraph = layerGraph(net0_dncnn.Layers);
    
    lgraph = replaceLayer(lgraph,'InputLayer',imageInputLayer(sizeIn,'Name','InputLayer','Normalization','none'));
    lgraph = replaceLayer(lgraph,'Conv20',convolution2dLayer(1,N_classes,'Name','Conv20'));
    lgraph = removeLayers(lgraph,'FinalRegressionLayer');
    lgraph = addLayers(lgraph,softmaxLayer('Name','Softmax'));
    
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_2_5'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_5_10'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_10_15'));
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_15_19'));
    
    lgraph = disconnectLayers(lgraph,'BNorm5','ReLU5');
    lgraph = disconnectLayers(lgraph,'BNorm10','ReLU10');
    lgraph = disconnectLayers(lgraph,'BNorm15','ReLU15');
    lgraph = disconnectLayers(lgraph,'BNorm19','ReLU19');
    
    lgraph = connectLayers(lgraph,'BNorm5','add_2_5/in1');
    lgraph = connectLayers(lgraph,'ReLU2','add_2_5/in2');
    lgraph = connectLayers(lgraph,'add_2_5','ReLU5');
    
    lgraph = connectLayers(lgraph,'BNorm10','add_5_10/in1');
    lgraph = connectLayers(lgraph,'ReLU5','add_5_10/in2');
    lgraph = connectLayers(lgraph,'add_5_10','ReLU10');
    
    lgraph = connectLayers(lgraph,'BNorm15','add_10_15/in1');
    lgraph = connectLayers(lgraph,'ReLU10','add_10_15/in2');
    lgraph = connectLayers(lgraph,'add_10_15','ReLU15');
    
    lgraph = connectLayers(lgraph,'BNorm19','add_15_19/in1');
    lgraph = connectLayers(lgraph,'ReLU15','add_15_19/in2');
    lgraph = connectLayers(lgraph,'add_15_19','ReLU19');
    
    lgraph = connectLayers(lgraph,'Conv20','Softmax');
    
%     lgraph = addLayers(lgraph,dicePixelClassificationLayer('Name','final_pxClasLayer'));
%     lgraph = connectLayers(lgraph,'Softmax','final_pxClasLayer');
    
    dlnet  = dlnetwork(lgraph);    
end