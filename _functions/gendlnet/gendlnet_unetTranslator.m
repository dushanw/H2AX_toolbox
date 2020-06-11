
function dlnet = gendlnet_unetTranslator(Nx,Nc)
        
    sizeIn = [Nx Nx Nc];
    
    numClasses = 2;
    encoderDepth = 3;
    lgraph = unetLayers(sizeIn,numClasses,'EncoderDepth',encoderDepth);
    lgraph = replaceLayer(lgraph,'ImageInputLayer',imageInputLayer(sizeIn,'Name','ImageInputLayer','Normalization','none'));
    lgraph = replaceLayer(lgraph,'Softmax-Layer',tanhLayer('Name','tanh1'));
    lgraph = replaceLayer(lgraph,'Final-ConvolutionLayer',convolution2dLayer(1,1,'Padding','same','Name','Final-ConvolutionLayer'));
    lgraph = removeLayers(lgraph,'Segmentation-Layer');
    
%     % convert LeakyRelu to leakyLeakyRelu
%     for i=1:19
%         lgraph = replaceLayer(lgraph,sprintf('ReLU%d',i),leakyReluLayer('Name',sprintf('LeakyRelu%d',i)));        
%     end

    dlnet  = dlnetwork(lgraph);
end