
function count = plt_segmentation(imds_I,imds_J,pxds_I,encoder,decoder,imgprocessor,savedir)
    
    %% I -> L  
    if ~isempty(imds_I)
        mkdir([savedir 'I_L/'])
        I_all           = imds_I.readall;

        count.img_name  = imds_I.Files;
        count.pred_I    = zeros(length(imds_I.Files),1);
        count.gt_I      = zeros(length(imds_I.Files),1);        
        
        if ~isempty(pxds_I)                     % read ground truth labels if available
            L_all = pxds_I.readall;
        end
        
        for i = 1:length(imds_I.Files)           
            img_name            = imds_I.Files{i};            
            img_name            = img_name(max(find(img_name=='/'))+1:end);      
            count.img_name{i,1} = img_name;
            
            dlI_now             = gpuArray(dlarray(I_all{i},'SSCB'));
            dlL_now             = predict(imgprocessor,dlI_now(:,:,1,:));
            
            J_now               = gather(dlI_now.extractdata);    
            L_now               = gather(dlL_now.extractdata);
            
            se = strel('disk',1);
            L_now = imopen(L_now(:,:,2),se)>0.7;
            
%            figure;
            hold off
            h = imshow(J_now(:,:,2),[]);
            set(gcf, 'Units', 'Inches', 'Position', [1,1,18,9])
            hold on
            
            %% plot ground truth            
            if ~isempty(pxds_I)             
                [L_temp,count.gt_I(i)] = bwlabel(L_all{i}=='fg');
                STATS     = regionprops(L_temp, 'Centroid');
                B         = bwboundaries(L_temp);    
         
                for k=1:count.gt_I(i)
                    c = round(STATS(k,1).Centroid(1));
                    r = round(STATS(k,1).Centroid(2));              
                    plot(c,r,'Or','MarkerSize',6,'LineWidth',1);                                  
                end
          
                if (size(B,1)>0)                
                    for k=1:size(B,1)
                        Outline = B{k};
                        plot(Outline(:,2),Outline(:,1),'r','LineWidth',1);              
                    end
                end          
            end
            
            
            %% plot predictions            
            [L_temp,count.pred_I(i)] = bwlabel(L_now);
            B         = bwboundaries(L_now);
            
            if size(J_now,3)==2 
              J_ki    = J_now(:,:,2);
              STATS   = regionprops(L_temp,J_ki,'Centroid','MeanIntensity');
              val_ki  = cat(1,STATS.MeanIntensity);
              ki_th   = mean(J_ki(:))*1.7;
            else
              STATS   = regionprops(L_temp, 'Centroid');
            end
            centroids = round(cat(1,STATS.Centroid));
            kiPos_idx = val_ki>ki_th;
            plot(centroids(:,1),centroids(:,2),'+g','MarkerSize',8,'LineWidth',1);
            plot(centroids(kiPos_idx,1),centroids(kiPos_idx,2),'Xr','MarkerSize',8,'LineWidth',1);
            
%             if (size(B,1)>0)
%                 for k=1:size(B,1)
%                     Outline = B{k};
%                     plot(Outline(:,2),Outline(:,1),'b','LineWidth',1);
%                     text(centroids(:,1),centroids(:,2),sprintf('%d',val_ki(k)))
%                 end
%             end
            
            hold off
            saveas(h,[savedir 'I_L/' img_name]);    
        end        
        struct2table(count)
        writetable(struct2table(count),[savedir '_counts.txt']);
    end
    
    
    if ~isempty(imds_J)
        mkdir([savedir 'J_L/'])
        J_all   = imds_J.readall;
        J_all   = (cat(4,J_all{:}));
        dlJ_all = dlarray(J_all, 'SSCB');
        
        for i = 1:size(dlJ_all,4)            
            img_name   = imds_J.Files{i};            
            img_name   = img_name(max(find(img_name=='/'))+1:end);      
            
            dlJ_now    = gpuArray(dlJ_all(:,:,:,i));
            dlIfake_now= predict(decoder,dlJ_now);
            dlL_now    = predict(imgprocessor,dlIfake_now);
            
            J_now      = gather(dlJ_now.extractdata);    
            L_now      = gather(dlL_now.extractdata);
            L_now      = L_now(:,:,2)>.5;        % threshold the channel 2 from the output   
        
            [L_temp,count.pred_J(i)] = bwlabel(L_now);
            STATS     = regionprops(L_temp, 'Centroid');
            B         = bwboundaries(L_now);
            figure;
            h = imshow(J_now,[]);
            hold on
            for k=1:count.pred_J(i)           
                c = round(STATS(k,1).Centroid(1));              
                r = round(STATS(k,1).Centroid(2));
                plot(c,r,'+g','MarkerSize',6,'LineWidth',1);                                  
            end
      
            if (size(B,1)>0)
                for k=1:size(B,1)
                    Outline = B{k};
                    plot(Outline(:,2),Outline(:,1),'b','LineWidth',1);
                end
            end 
            hold off
            saveas(h,[savedir 'J_L/' img_name]);
        end
    end
       
    
    
end
    