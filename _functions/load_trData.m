% 20200609 by Dushan N. Wadduwage

function trData = load_trData(pram)

  imds_gt       = imageDatastore(pram.dir_imds_gt_0,'ReadFcn',@(fname)ds_imread(fname,pram));
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
    L(:,:,1)    = ~ (I_seg_hand{i}(:,:,1)==255);
    L(:,:,2)    = ~ L(:,:,1) & ~(I_seg_algo{i}(:,:,1)==255);
    
    % L    = L*2-1;                                           % off-label is -1
    
    % un-augmented
    [I_tr_temp L_tr_temp] = get_sub_images(I,L,pram.Nx); 
    I_tr = cat(4,I_tr,I_tr_temp);
    L_tr = cat(4,L_tr,L_tr_temp);
    
    % horz-flipp
    I    = flip(I,1);
    L    = flip(L,1);
    [I_tr_temp L_tr_temp] = get_sub_images(I,L,pram.Nx); 
    I_tr = cat(4,I_tr,I_tr_temp);
    L_tr = cat(4,L_tr,L_tr_temp);
    
    % vert-flipp
    I    = flip(I,2);
    L    = flip(L,2);
    [I_tr_temp L_tr_temp] = get_sub_images(I,L,pram.Nx); 
    I_tr = cat(4,I_tr,I_tr_temp);
    L_tr = cat(4,L_tr,L_tr_temp);
    
    % rotate90, no-flip
    I    = rot90(I);
    L    = rot90(L);
    [I_tr_temp L_tr_temp] = get_sub_images(I,L,pram.Nx); 
    I_tr = cat(4,I_tr,I_tr_temp);
    L_tr = cat(4,L_tr,L_tr_temp);
        
    % horz-flipp
    I    = flip(I,1);
    L    = flip(L,1);
    [I_tr_temp L_tr_temp] = get_sub_images(I,L,pram.Nx); 
    I_tr = cat(4,I_tr,I_tr_temp);
    L_tr = cat(4,L_tr,L_tr_temp);
    
    % vert-flipp
    I    = flip(I,2);
    L    = flip(L,2);
    [I_tr_temp L_tr_temp] = get_sub_images(I,L,pram.Nx); 
    I_tr = cat(4,I_tr,I_tr_temp);
    L_tr = cat(4,L_tr,L_tr_temp);    
  end    
  b_shf  = randperm(size(I_tr,4));
  I_tr   = I_tr(:,:,:,b_shf);
  L_tr   = L_tr(:,:,:,b_shf);
  
  trData.I_tr = I_tr(:,:,:,1:round(0.95*size(I_tr,4)));
  trData.L_tr = L_tr(:,:,:,1:round(0.95*size(I_tr,4)));
  trData.I_vl = I_tr(:,:,:,round(0.95*size(I_tr,4))+1:end);
  trData.L_vl = L_tr(:,:,:,round(0.95*size(I_tr,4))+1:end);
  
end

function [I_tr L_tr] = get_sub_images(I,L,Nx) 
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










