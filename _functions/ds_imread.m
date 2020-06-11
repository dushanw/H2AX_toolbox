

function I = ds_imread(img_fname,pram)                
    
    I0  = imread(img_fname);
    I0  = I0(:,:,pram.ds_imread_channedls);
    I   = single(I0);

%    I   = single(I0(1:2000,1:2000,:));
    
    switch pram.ds_imread_channedls_scale_method
        case 'rescale10k-zerocenter'
            I = I/1e4;  
            I = I - mean(I(:));
        case 'rescale10k'            
            I = I/1e4;        
        case 'rescale10k-noised'            
            I = I/1e4;        
            I = poissrnd(I*100)/100;            
        case 'rescale10k-4xlowres'            
            I = I/1e4;   
            I = imresize(imresize(I,0.25),4);            
        case 'rescale_m1_p1'            
            I = (I/2^16 - 0.5)/0.5;   
        case 'rescale_m1_p1-4xlowres'            
            I = imresize(imresize(I,0.25),4);
            I = (I/2^16 - 0.5)/0.5;            
        case 'rescale1k'            
            I = I/1e3;        
        case 'max-zerocenter'
            I = I/max(I(:));
            I = I - 0.5;
        case 'max'
            I = I/max(I(:));
        case 'mean-zerocenter'  
            I = I-mean(I(:));
            I = I/(3*std(I(:)));    
        case 'mean-histeq'  
            I = single(histeq(I0,2^16));
            I = I/mean(I(:));
        case 'mean'
            I = I/mean(I(:));            
        case 'log'
            I = log(I);
        case 'none'
            I = I;
    end
end