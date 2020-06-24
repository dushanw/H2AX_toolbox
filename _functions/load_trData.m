% 20200609 by Dushan N. Wadduwage

function trData = load_trData(pram)

%% load and augment exp data
  if pram.is_imgTranslator_on
    imds_gt   = imageDatastore(pram.dir_imds_gt_0, 'ReadFcn',@(fname)ds_imread(fname,pram),'FileExtensions',pram.ds_imread_extensions);
    imds_exp  = imageDatastore(pram.dir_imds_exp_0,'ReadFcn',@(fname)ds_imread(fname,pram),'FileExtensions',pram.ds_imread_extensions);
        
    % select channel
    imds_gt   = subset(imds_gt ,find(contains(imds_gt.Files , pram.ds_imread_chText))); % ex for pram.ds_imread_chText = '_w4' (for dapi)
    imds_exp  = subset(imds_exp,find(contains(imds_exp.Files, pram.ds_imread_chText)));
    
    parfor i=1:length(imds_gt.Files)    
      I_gt_tr{i} = single([]);
      I_gt_tr{i} = patch_and_augment(readimage(imds_gt,i),I_gt_tr{i},pram); 
    end
    I_gt_tr = cat(4,I_gt_tr{:});
    
    parfor i=1:length(imds_exp.Files)  
      I_exp_tr{i} = single([]);
      I_exp_tr{i} = patch_and_augment(imresize(readimage(imds_exp,i),pram.rsf),...
                                     I_exp_tr{i},pram); 
    end
    I_exp_tr = cat(4,I_exp_tr{:});

    I_gt_tr   = I_gt_tr(:,:,:,randperm(size(I_gt_tr,4)));
    I_exp_tr  = I_exp_tr(:,:,:,randperm(size(I_exp_tr,4)));

    I_gt_tr   = I_gt_tr (:,:,:,1:min([size(I_gt_tr,4), size(I_exp_tr,4), pram.max_tr_size]));
    I_exp_tr  = I_exp_tr(:,:,:,1:min([size(I_gt_tr,4), size(I_exp_tr,4), pram.max_tr_size]));

    trData.I_gt_vl  = I_gt_tr (:,:,:,round(0.95*size(I_gt_tr,4))+1:end);
    trData.I_exp_vl = I_exp_tr(:,:,:,round(0.95*size(I_exp_tr,4))+1:end);        
    trData.I_gt_tr  = I_gt_tr (:,:,:,1:round(0.95*size(I_gt_tr,4)));
    trData.I_exp_tr = I_exp_tr(:,:,:,1:round(0.95*size(I_exp_tr,4)));
  end

%% load and augment gt and segmentation data
  if pram.is_imgProcessor_on
    imds_gt       = imageDatastore(pram.dir_imds_gt_0,'ReadFcn',@(fname)ds_imread(fname,pram),'FileExtensions',pram.ds_imread_extensions);
    imds_seg_hand = imageDatastore(pram.dir_pxds_gt_hand_0);
    imds_seg_algo = imageDatastore(pram.dir_pxds_gt_algo_0);

    I_gt_all      = imds_gt.readall;
    I_seg_hand    = imds_seg_hand.readall;
    I_seg_algo    = imds_seg_algo.readall;

    I_tr = single([]);
    L_tr = single([]);

    for i=1:length(I_gt_all)    
      clear I L    
      I           = I_gt_all{i};
      L(:,:,1)    = I_seg_hand{i}(:,:,1) > (single(max(I_seg_hand{i}(:)))/2);            
      L(:,:,2)    = ~ L(:,:,1) & I_seg_algo{i}(:,:,1) > (single(max(I_seg_algo{i}(:)))/2);
      
%       se          = strel('disk',1);
%       L_edge      = imdilate(L(:,:,1),se) & ~L(:,:,1);
%       L(:,:,2)    = L(:,:,2) | L_edge;  
      
      [I_tr L_tr] = patch_and_augment_paired(I,L,I_tr,L_tr,pram);      
    end    
    b_shf  = randperm(size(I_tr,4));
    I_tr   = I_tr(:,:,:,b_shf);
    L_tr   = L_tr(:,:,:,b_shf);

    trData.I_vl = I_tr(:,:,:,round(0.95*size(I_tr,4))+1:end);
    trData.L_vl = L_tr(:,:,:,round(0.95*size(I_tr,4))+1:end);
    trData.I_tr = I_tr(:,:,:,1:round(0.95*size(I_tr,4)));
    trData.L_tr = L_tr(:,:,:,1:round(0.95*size(I_tr,4)));
  end
  
end


function [I_tr L_tr] = patch_and_augment_paired(I,L,I_tr,L_tr,pram)
    % un-augmented
    [I_tr_temp L_tr_temp] = get_sub_images_paired(I,L,pram.Nx); 
    I_tr = cat(4,I_tr,I_tr_temp);
    L_tr = cat(4,L_tr,L_tr_temp);

    if pram.ds_imread_augment
      % horz-flipp
      I    = flip(I,1);
      L    = flip(L,1);
      [I_tr_temp L_tr_temp] = get_sub_images_paired(I,L,pram.Nx); 
      I_tr = cat(4,I_tr,I_tr_temp);
      L_tr = cat(4,L_tr,L_tr_temp);

      % vert-flipp
      I    = flip(I,2);
      L    = flip(L,2);
      [I_tr_temp L_tr_temp] = get_sub_images_paired(I,L,pram.Nx); 
      I_tr = cat(4,I_tr,I_tr_temp);
      L_tr = cat(4,L_tr,L_tr_temp);

      % rotate90, no-flip
      I    = rot90(I);
      L    = rot90(L);
      [I_tr_temp L_tr_temp] = get_sub_images_paired(I,L,pram.Nx); 
      I_tr = cat(4,I_tr,I_tr_temp);
      L_tr = cat(4,L_tr,L_tr_temp);

      % horz-flipp
      I    = flip(I,1);
      L    = flip(L,1);
      [I_tr_temp L_tr_temp] = get_sub_images_paired(I,L,pram.Nx); 
      I_tr = cat(4,I_tr,I_tr_temp);
      L_tr = cat(4,L_tr,L_tr_temp);

      % vert-flipp
      I    = flip(I,2);
      L    = flip(L,2);
      [I_tr_temp L_tr_temp] = get_sub_images_paired(I,L,pram.Nx); 
      I_tr = cat(4,I_tr,I_tr_temp);
      L_tr = cat(4,L_tr,L_tr_temp);   
    end
end

function [I_tr L_tr] = patch_and_augment(I,I_tr,pram)
  % un-augmented
  I_tr_temp = get_sub_images(I,pram.Nx); 
  I_tr      = cat(4,I_tr,I_tr_temp);
  
  inds = find(squeeze(var(I_tr,0,[1 2]))>pram.ds_imread_varTh);
  I_tr      = I_tr(:,:,:,inds);
  
  if pram.ds_imread_augment
    % horz-flipp
    I         = flip(I,1);
    I_tr_temp = get_sub_images(I,pram.Nx); 
    I_tr      = cat(4,I_tr,I_tr_temp);

    % vert-flipp
    I    = flip(I,2);
    I_tr_temp = get_sub_images(I,pram.Nx); 
    I_tr      = cat(4,I_tr,I_tr_temp);

    % rotate90, no-flip
    I    = rot90(I);
    I_tr_temp = get_sub_images(I,pram.Nx); 
    I_tr      = cat(4,I_tr,I_tr_temp);

    % horz-flipp
    I    = flip(I,1);
    I_tr_temp = get_sub_images(I,pram.Nx); 
    I_tr      = cat(4,I_tr,I_tr_temp);

    % vert-flipp
    I    = flip(I,2);
    I_tr_temp = get_sub_images(I,pram.Nx); 
    I_tr      = cat(4,I_tr,I_tr_temp);
  end
end

function [I_tr L_tr] = get_sub_images_paired(I,L,Nx) 
  kk = 1;
  I_tr = [];
  L_tr = [];
  for j=1:Nx:size(I,1)-Nx
    for k=1:Nx:size(I,2)-Nx          
        I_tr(:,:,1,kk) = I(j:j+Nx-1,k:k+Nx-1,1);                     
        L_tr(:,:,:,kk) = L(j:j+Nx-1,k:k+Nx-1,:); 
        kk = kk+1;
    end
  end
end

function [I_tr L_tr] = get_sub_images(I,Nx) 
  kk = 1;
  I_tr = [];
  for j=1:Nx:size(I,1)-Nx
    for k=1:Nx:size(I,2)-Nx          
        I_tr(:,:,1,kk) = I(j:j+Nx-1,k:k+Nx-1,1);                     
        kk = kk+1;
    end
  end
end










