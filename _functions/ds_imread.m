

function I = ds_imread(img_fname,pram)                

  switch pram.ds_imread_read_method
    case 'bfopen_3D2maxproj-rescale10k'
      V0= bfopen(img_fname);
      I = single(max(cat(3,V0{1}{:,1}),[],3));
      % hack
      I = imresize(I,0.25);
      % ----
      I = I/1e4;
    case 'bfopen_3D2maxproj-max-zerocenter'
      V0 = tiffread2(img_fname);
      I = single(max(cat(3,V0.data),[],3));
      % hack
      % I = imresize(I,0.25);
      % ----        
      I = I/max(I(:));
      I = 2*I - 1;
    case 'rescale10k-zerocenter'
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I/1e4;  
      I = I - mean(I(:));
    case 'rescale10k'    
      for i=1:pram.ds_imread_channedls
        I0(:,:,i)= imread(img_fname,i);
      end
      I = single(I0);
      I = I/1e4;        
    case 'rescale10k-noised'            
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I/1e4;        
      I = poissrnd(I*100)/100;            
    case 'rescale10k-4xlowres'            
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I/1e4;   
      I = imresize(imresize(I,0.25),4);            
    case 'rescale_m1_p1'            
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = (I/2^16 - 0.5)/0.5;   
    case 'rescale_m1_p1-4xlowres'            
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = imresize(imresize(I,0.25),4);
      I = (I/2^16 - 0.5)/0.5;            
    case 'rescale1k'            
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I/1e3;        
    case 'max-zerocenter'
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I/max(I(:));
      I = I - 0.5;
    case 'max'
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I/max(I(:));
    case 'mean-zerocenter'  
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I-mean(I(:));
      I = I/(3*std(I(:)));    
    case 'mean-histeq'  
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = single(histeq(I0,2^16));
      I = I/mean(I(:));
    case 'mean'
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = I/mean(I(:));            
    case 'log'
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
      I = log(I);
    case 'none'
      I0= imread(img_fname);
      I = single(I0(:,:,pram.ds_imread_channedls));
  end
end