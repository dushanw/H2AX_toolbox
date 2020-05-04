

function plt_translation(imds_I,imds_J,encoder,decoder,savedir)

    
    %% I -> J-fake -> I-fakefake
    if ~isempty(imds_I)
        mkdir([savedir 'I_fakefake/'])
        I_all   = imds_I.readall;
        I_all   = (cat(4,I_all{:}));
        dlI_all = dlarray(I_all, 'SSCB');
        for i = 1:size(dlI_all,4)
            img_neame       = imds_I.Files{i};
            img_neame       = img_neame(max(find(img_neame=='/'))+1:end);       
            
            dlI_now         = gpuArray(dlI_all(:,:,:,i));
            dlJfake_now     = predict(encoder,dlI_now);
            dlIfakefake_now = predict(decoder,dlJfake_now);

            figure('units','normalized','outerposition',[0 0 1 1])
            imagesc([rescale(dlI_now.extractdata) ...
                     rescale(dlJfake_now.extractdata) ...
                     rescale(dlIfakefake_now.extractdata)]); axis image

            saveas(gca,sprintf('%splt_%d_%s.jpeg',[savedir 'I_fakefake/'],i,img_neame));  
            close all
            disp(sprintf('translated: I->Jfake->Ifakefake - %s (%d/%d)',img_neame,i,size(dlI_all,4)))
        end        
    end
    
    %% J -> I-fake -> J-fakefake
    if ~isempty(imds_J)                
        mkdir([savedir 'J_fakefake/'])
        J_all   = imds_J.readall;
        J_all   = (cat(4,J_all{:}));
        dlJ_all = dlarray(J_all, 'SSCB');
        for i = 1:size(dlJ_all,4)
            img_neame       = imds_J.Files{i};
            img_neame       = img_neame(max(find(img_neame=='/'))+1:end);       
            
            dlJ_now         = gpuArray(dlJ_all(:,:,:,i));
            dlIfake_now     = predict(decoder,dlJ_now);
            dlJfakefake_now = predict(encoder,dlIfake_now);

            figure('units','normalized','outerposition',[0 0 1 1])
            imagesc([rescale(dlJ_now.extractdata) ...
                     rescale(dlIfake_now.extractdata) ...
                     rescale(dlJfakefake_now.extractdata)]); axis image
            saveas(gca,sprintf('%splt_%d_%s.jpeg',[savedir 'J_fakefake/'],i,img_neame));  
            close all
            disp(sprintf('translated: J->Ifake->Jfakefake - %s (%d/%d)',img_neame,i,size(dlJ_all,4)))
        end  
    end

end