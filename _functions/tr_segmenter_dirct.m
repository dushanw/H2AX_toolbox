% 20191123 by Dushan N. Wadduwage (wadduwage@fas.harvard.edu)
% 20200429 modified by Dushan N. Wadduwage (wadduwage@fas.harvard.edu)
% segmenter



% function net = tr_segmenter_dirct(lgraph,imds_gt,pxds_gt,pram)
%     
%    %% network parameters 
%     patchSize       = [pram.Nx pram.Nx];
%     miniBatchSize   = 64;
%     numClasses      = pram.N_classes;
%     classNames      = pram.classNames;
%     pixelLabelIDs   = pram.pxLblIds;
% 
%     %% set other datastore parameters
%     I_temp          = imds_gt.read;     
%     imds_gt.reset;    
%     N_PatchesPerImg = prod(size(I_temp)./patchSize)*16;
%     
%     %% image-patch datastores
%     patchds_tr                  = randomPatchExtractionDatastore(imds_gt,pxds_gt,patchSize,'PatchesPerImage',N_PatchesPerImg);
%     patchds_tr.MiniBatchSize    = miniBatchSize;
%%
function net = tr_segmenter_dirct(lgraph,trData,pram)

    %% train network
    l2reg = 0.0001;
    options = trainingOptions('sgdm', ...
        'Momentum',0.9, ...
        'InitialLearnRate',pram.learnRateImgprocessor, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',pram.learnRateDropInterval, ...
        'LearnRateDropFactor',pram.learnRateDropFactor, ...
        'L2Regularization',l2reg, ...
        'MaxEpochs',pram.numEpochs ,...        
        'ValidationFrequency', 500, ...
        'MiniBatchSize',pram.miniBatchSize, ...
        'GradientThresholdMethod','l2norm', ...
        'Plots','training-progress', ...
        'GradientThreshold',0.01,...
        'ExecutionEnvironment','multi-gpu');                % 'ValidationData',patchds_val,...
    
%     X = trData.I_tr;
%     Y = cat(3,cat(3,trData.L_tr(:,:,1,:)*-1+1,trData.L_tr(:,:,1:2,:))); 
%     A = zeros(size(Y,1),size(Y,2),1,size(Y,4));
%     A(Y(:,:,1,:)==1)=0;  
%     A(Y(:,:,2,:)==1)=1;  
%     A(Y(:,:,3,:)==1)=2;  
%     
%     mkdir('./_tempData/In/');
%     mkdir('./_tempData/Out/');
%     for i=1:size(X,4)
%       i
%       I = X(:,:,:,i);
%       L = uint8(A(:,:,:,i));
%       save(sprintf('./_tempData/In/img_%0.6d.mat' ,i),'I');
%       imwrite(L,sprintf('./_tempData/Out/pxd_%0.6d.tif',i));
%     end
    imds    = imageDatastore('./_tempData/In/','ReadFcn',@loadDotMatImages,'FileExtensions','.mat');
    pxds    = pixelLabelDatastore('./_tempData/Out/', pram.classNames,pram.pxLblIds);
    pximds  = pixelLabelImageDatastore(imds,pxds);
    
    net = trainNetwork(pximds,layerGraph(lgraph),options);
end

function I = loadDotMatImages(fname)
    load(fname)    
end



