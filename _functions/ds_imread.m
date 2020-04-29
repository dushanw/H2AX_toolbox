

function I = ds_imread(img_fname,pram)                
    
    I0  = imread(img_fname);
    I   = single(I0(:,:,pram.ds_imread_channedls));
    
    switch pram.ds_imread_channedls_scale_method
        case 'rescale10k'            
            I = I/1e4;        
        case 'rescale1k'            
            I = I/1e3;        
        case 'max'
            I = I/max(I(:));
        case 'mean'
            I = I/mean(I(:));
        case 'none'
            I = I;
    end
end