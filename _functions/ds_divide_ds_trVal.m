
function [ds_tr ds_val] = ds_divide_ds_trVal(ds,N_files_val)
    % first N_files_val are assigned to validation ds, rest are put in training ds
    N_files         = length(ds.Files)
    
    ds_val          = ds.subset(1            :N_files_val);
    ds_tr           = ds.subset(N_files_val+1:N_files    );
end