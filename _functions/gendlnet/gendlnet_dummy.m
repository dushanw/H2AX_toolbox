
function dlnet = gendlnet_dummy(Nx,Nc)

    sizeIn  = [Nx Nx Nc];
    layers  = [
                imageInputLayer(sizeIn,'Name','InputLayer','Normalization','none')
               ];
    
    lgraph  = layerGraph(layers);
    dlnet   = dlnetwork(lgraph);    
end