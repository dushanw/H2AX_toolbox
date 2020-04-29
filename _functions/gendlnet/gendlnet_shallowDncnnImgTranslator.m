
function dlnet = gendlnet_shallowDncnnImgTranslator(Nx,Nc)
        
    sizeIn = [Nx Nx Nc];

    net0_dncnn = denoisingNetwork('dncnn');
    lgraph = layerGraph(net0_dncnn.Layers);
    
    lgraph = replaceLayer(lgraph,'Conv1',convolution2dLayer(5,64,'Padding',[2 2 2 2],'Name','Conv1'));
    lgraph = replaceLayer(lgraph,'InputLayer',imageInputLayer(sizeIn,'Name','InputLayer','Normalization','none'));
    lgraph = removeLayers(lgraph,'FinalRegressionLayer');

    for i=6:19
        lgraph = removeLayers(lgraph,sprintf('Conv%d',i));
        lgraph = removeLayers(lgraph,sprintf('BNorm%d',i));
        lgraph = removeLayers(lgraph,sprintf('ReLU%d',i));
    end
    lgraph = addLayers(lgraph,additionLayer(2,'Name','add_2_5'));
    
    lgraph = disconnectLayers(lgraph,'BNorm5','ReLU5');
    lgraph = connectLayers(lgraph,'BNorm5','add_2_5/in1');
    lgraph = connectLayers(lgraph,'ReLU2','add_2_5/in2');
    lgraph = connectLayers(lgraph,'add_2_5','ReLU5');
    lgraph = connectLayers(lgraph,'ReLU5','Conv20');
    
    dlnet  = dlnetwork(lgraph);
end