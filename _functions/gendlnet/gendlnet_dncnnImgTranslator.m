
function dlnet = gendlnet_dncnnImgTranslator(Nx,Nc)
        
    sizeIn = [Nx Nx Nc];

    net0_dncnn = denoisingNetwork('dncnn');
    lgraph = layerGraph(net0_dncnn.Layers);
    
    lgraph = replaceLayer(lgraph,'Conv1',convolution2dLayer(5,64,'Padding',[2 2 2 2],'Name','Conv1'));
    lgraph = replaceLayer(lgraph,'InputLayer',imageInputLayer(sizeIn,'Name','InputLayer','Normalization','none'));
    lgraph = removeLayers(lgraph,'FinalRegressionLayer');
    
    % convert LeakyRelu to leakyLeakyRelu
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
    
    dlnet = dlnetwork(lgraph);
end