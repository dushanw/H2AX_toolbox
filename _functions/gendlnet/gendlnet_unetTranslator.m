
function dlnet = gendlnet_unetTranslator(Nx,Nc)
        
  sizeIn = [Nx Nx Nc];

  numClasses = 2;
  encoderDepth = 3;
  lgraph = unetLayers(sizeIn,numClasses,'EncoderDepth',encoderDepth);
  lgraph = replaceLayer(lgraph,'ImageInputLayer',imageInputLayer(sizeIn,'Name','ImageInputLayer','Normalization','none'));
  lgraph = replaceLayer(lgraph,'Softmax-Layer',tanhLayer('Name','tanh1'));
  lgraph = replaceLayer(lgraph,'Final-ConvolutionLayer',convolution2dLayer(1,1,'Padding','same','Name','Final-ConvolutionLayer'));
  lgraph = removeLayers(lgraph,'Segmentation-Layer');

  % convert LeakyRelu to leakyLeakyRelu
  for i=1:encoderDepth
    lgraph = replaceLayer(lgraph,sprintf('Encoder-Stage-%d-ReLU-1',i),leakyReluLayer('Name',sprintf('Encoder-Stage-%d-LeakyReLU-1',i)));        
    lgraph = replaceLayer(lgraph,sprintf('Encoder-Stage-%d-ReLU-2',i),leakyReluLayer('Name',sprintf('Encoder-Stage-%d-LeakyReLU-2',i)));        
    lgraph = replaceLayer(lgraph,sprintf('Decoder-Stage-%d-UpReLU',i),leakyReluLayer('Name',sprintf('Decoder-Stage-%d-UpLeakyReLU',i)));
    lgraph = replaceLayer(lgraph,sprintf('Decoder-Stage-%d-ReLU-1',i),leakyReluLayer('Name',sprintf('Decoder-Stage-%d-LeakyReLU-1',i)));
    lgraph = replaceLayer(lgraph,sprintf('Decoder-Stage-%d-ReLU-2',i),leakyReluLayer('Name',sprintf('Decoder-Stage-%d-LeakyReLU-2',i)));
  end
  lgraph = replaceLayer(lgraph,'Bridge-ReLU-1',leakyReluLayer('Name','Bridge-LeakyReLU-1'));        
  lgraph = replaceLayer(lgraph,'Bridge-ReLU-2',leakyReluLayer('Name','Bridge-LeakyReLU-2'));        
  
  dlnet  = dlnetwork(lgraph);
end