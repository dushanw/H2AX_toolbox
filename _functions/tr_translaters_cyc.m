% I - gt style images
% J - exp style images
% J_fake = enc(I);
% I_fake = dec(J);
% I_fake_fake = dec(J_fake)
% J_fake_fake = dec(I_fake)

function [encoder decoder D_I D_J tr_info] = tr_translaters_cyc ...
         (encoder,decoder,D_I,D_J,imds_I,imds_J,pram)

    %% network parameters 
    patchSize       = [pram.Nx pram.Nx];
    miniBatchSize   = pram.miniBatchSize;

    %% set other datastore parameters
    I_temp          = imds_I.read;     
    imds_I.reset;    
    N_PatchesPerImg = round(prod(size(I_temp)./patchSize)*16);
    
    %% divide datastores training and validation
    imds_I                      = shuffle(imds_I);
    imds_J                      = shuffle(imds_J);
    [imds_I_tr imds_I_val]      = ds_divide_ds_trVal(imds_I,1);
    [imds_J_tr imds_J_val]      = ds_divide_ds_trVal(imds_J,1);
    
    N_files                     = min(length(imds_I_tr.Files),length(imds_J_tr.Files));
    
    %% validation data
    patchds_val                 = randomPatchExtractionDatastore(imds_I_val,...
                                                                 imds_J_val,...
                                                                 patchSize,'PatchesPerImage',N_PatchesPerImg);
    patchds_val.MiniBatchSize   = miniBatchSize;
    
    data_val        = read(patchds_val);
    I_val           = cat(4,data_val.InputImage{:});
    J_val           = cat(4,data_val.ResponseImage{:});

    dlI_val         = dlarray(I_val, 'SSCB');
    dlJ_val         = dlarray(J_val, 'SSCB');

    if (pram.executionEnvironment == "auto" && canUseGPU) || pram.executionEnvironment == "gpu"
        dlI_val     = gpuArray(dlI_val);
        dlJ_val     = gpuArray(dlJ_val);
    end
    
    
    %% training loop
    figure
    iteration       = 0;
    start           = tic;
    
    for i = 1:pram.numEpochs
%         imds_I_tr = shuffle(imds_I_tr);
%         imds_J_tr = shuffle(imds_J_tr);
        % generate an image-patch datastores with randomly drawn equal number of images form imds_I and imds_J
        patchds_tr                  = randomPatchExtractionDatastore(imds_I_tr.subset(1:N_files),...
                                                                     imds_J_tr.subset(1:N_files),...
                                                                     patchSize,'PatchesPerImage',N_PatchesPerImg);
        patchds_tr.MiniBatchSize    = miniBatchSize;
        patchds_tr = shuffle(patchds_tr);
%      for i = 1:pram.numEpochs        
%         % Reset and shuffle datastore.
%         reset(patchds_tr);
        
                   
%         data_val         = read(patchds_tr);
%         I_val        = cat(4,data_val.InputImage{:});
%         J_val        = cat(4,data_val.ResponseImage{:});
% 
%         dlI_val      = dlarray(I_val, 'SSCB');
%         dlJ_val      = dlarray(J_val, 'SSCB');
% 
%         if (pram.executionEnvironment == "auto" && canUseGPU) || pram.executionEnvironment == "gpu"
%             dlI_val      = gpuArray(dlI_val);
%             dlJ_val      = gpuArray(dlJ_val);
%         end


        while hasdata(patchds_tr)
            iteration   = iteration + 1;

            data_tr         = read(patchds_tr);
            if size(data_tr,1)<pram.miniBatchSize
                break
            end
            
            I_tr        = cat(4,data_tr.InputImage{:});
            J_tr        = cat(4,data_tr.ResponseImage{:});

            dlI_tr      = dlarray(I_tr, 'SSCB');
            dlJ_tr      = dlarray(J_tr, 'SSCB');
            
            if (pram.executionEnvironment == "auto" && canUseGPU) || pram.executionEnvironment == "gpu"
                dlI_tr      = gpuArray(dlI_tr);
                dlJ_tr      = gpuArray(dlJ_tr);
            end
            
            % Evaluate the model gradients and the generator state
            [gradients_encoder,...
             gradients_decoder,...
             gradients_D_I,...
             gradients_D_J,...
             state_encoder,...             
             state_decoder,...             
             losses_itr] = dlfeval(@f_modelGradients,...
                                    encoder, decoder, D_I, D_J,...
                                    dlI_tr, dlJ_tr,...
                                    i, iteration,pram);

            encoder.State      = state_encoder;
            decoder.State      = state_decoder;

            allLosses(iteration,:)  = extractdata(losses_itr);  % track losses
            
            % Update the _D_I network parameters.
            [D_I.Learnables,pram.trailingAvg_D_I,pram.trailingAvgSq_D_I] = ...
                adamupdate(D_I.Learnables, gradients_D_I, ...
                           pram.trailingAvg_D_I , pram.trailingAvgSq_D_I, iteration, ...
                           pram.learnRateDiscriminator, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);
            % Update the _D_J network parameters.
            [D_J.Learnables,pram.trailingAvg_D_J,pram.trailingAvgSq_D_J] = ...
                adamupdate(D_J.Learnables, gradients_D_J, ...
                           pram.trailingAvg_D_J , pram.trailingAvgSq_D_J, iteration, ...
                           pram.learnRateDiscriminator, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);                
            % Update the encoder network parameters.
            [encoder.Learnables,pram.trailingAvg_encoder,pram.trailingAvgSq_encoder] = ...
                adamupdate(encoder.Learnables, gradients_encoder, ...
                           pram.trailingAvg_encoder, pram.trailingAvgSq_encoder, iteration, ...
                           pram.learnRate_encoder, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);
            % Update the encoder network parameters.
            [decoder.Learnables,pram.trailingAvg_decoder,pram.trailingAvgSq_decoder] = ...
                adamupdate(decoder.Learnables, gradients_decoder, ...
                           pram.trailingAvg_decoder, pram.trailingAvgSq_decoder, iteration, ...
                           pram.learnRate_decoder, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);
                    
            % Every 20 iterations, display validation results                    
            if mod(iteration,20) == 0 || iteration == 1                           
                dlI_ori     = dlI_val(:,:,:,randi(size(dlI_val,4)));
                dlJ_ori     = dlJ_val(:,:,:,randi(size(dlI_val,4)));
                dlJ_fake    = predict(encoder,dlI_ori);
                dlI_recon   = predict(decoder,dlJ_fake);
                dlI_fake    = predict(decoder,dlJ_ori);
                dlJ_recon   = predict(encoder,dlI_fake);
                
                img_I       = (imtile(cat(1, extractdata(dlI_ori),extractdata(dlJ_fake),extractdata(dlI_recon) ),'GridSize',[1 1]));
                img_J       = (imtile(cat(1, extractdata(dlJ_ori),extractdata(dlI_fake),extractdata(dlJ_recon) ),'GridSize',[1 1]));
                
                subplot(1,3,1);imagesc([img_I img_J]);axis image;axis off;colorbar
                subplot(1,3,2);plot(allLosses(:,1:4));
                                    legend('L Enc','L Dec','L D I','L D J') 
                subplot(1,3,3);plot(allLosses(:,5:6));
                                    legend('Lcyc','Lid')                                 

                                    
                % Update the title with training progress information.
                t_duration      = duration(0,0,toc(start),'Format','hh:mm:ss');
                title(...
                    "Epoch: " + i + ", " + ...
                    "Iteration: " + iteration + ", " + ...
                    "Elapsed: " + string(t_duration))
                drawnow
            end            
        end
    end
    
    allLosses           = gather(allLosses);  
    tr_info.loss_encoder   = allLosses(:,1);
    tr_info.loss_decoder   = allLosses(:,2);
    tr_info.loss_D_I       = allLosses(:,3);
    tr_info.loss_D_J       = allLosses(:,4);    
end


%% model gradient function
function    [gradients_encoder,...
             gradients_decoder,...
             gradients_D_I,...
             gradients_D_J,...
             state_encoder,...             
             state_decoder,...             
             losses_itr] = f_modelGradients(encoder, decoder, D_I, D_J,...
                                            dlI, dlJ,...
                                            epoch, iteration, pram)

    % calculate translations and predictions from the discriminators
    [dlJ_fake   state_encoder]  = forward(encoder,dlI);
    [dlI_fake   state_decoder]  = forward(decoder,dlJ);
    [dlJ_recon  state_encoder]  = forward(encoder,dlI_fake);
    [dlI_recon  state_decoder]  = forward(decoder,dlJ_fake);
    [dlJ_id     state_encoder]  = forward(encoder,dlJ);
    [dlI_id     state_decoder]  = forward(decoder,dlI);

    
    dlI_pred            = forward(D_I, dlI);
    dlJ_pred            = forward(D_J, dlJ);
    dlI_fake_pred       = forward(D_I, dlI_fake);
    dlJ_fake_pred       = forward(D_J, dlJ_fake);
    dlI_recon_pred      = forward(D_I, dlI_recon);    
    dlJ_recon_pred      = forward(D_J, dlJ_recon);    

    % Calculate the GAN loss
    [loss_encoder, loss_decoder, loss_D_I, loss_D_J, loss_cyc, loss_id] = ...
                                                                 f_ganLoss(dlI_pred,...
                                                                 dlJ_pred,...
                                                                 dlI_fake_pred,...
                                                                 dlJ_fake_pred,...
                                                                 dlI,dlI_recon,dlI_id,...
                                                                 dlJ,dlJ_recon,dlJ_id,...
                                                                 pram.gammaCyc);

    disp(sprintf('%d %d\t L_enc&dec %d\t L_cyc %d \t L_id %d \t L_DI %d \t L_DJ %d',...
                epoch,iteration, loss_encoder, loss_cyc, loss_id, loss_D_I, loss_D_J));    
    
    losses_itr = [loss_encoder, loss_decoder, loss_D_I, loss_D_J, loss_cyc, loss_id];

    % For each network, calculate the gradients with respect to the loss.
    gradients_encoder   = dlgradient(loss_encoder, encoder.Learnables,'RetainData',true);    
    gradients_decoder   = dlgradient(loss_decoder, decoder.Learnables,'RetainData',true);    
    gradients_D_I       = dlgradient(loss_D_I    , D_I.Learnables);
    gradients_D_J       = dlgradient(loss_D_J    , D_J.Learnables);
end


function [loss_encoder, loss_decoder, loss_D_I, loss_D_J, loss_cyc, loss_id] = ...
                                                                      f_ganLoss(dlI_pred,...
                                                                      dlJ_pred,...
                                                                      dlI_fake_pred,...
                                                                      dlJ_fake_pred,...
                                                                      dlI,dlI_recon,dlI_id,...
                                                                      dlJ,dlJ_recon,dlJ_id,...
                                                                      gamma)
    loss_mode = 'softLabels-noLog';
    switch loss_mode
        case 'slack'                                                                  
            delta = 1e-3;                                                           % slack value
            loss_D_I_real   = -mean(log(delta+sigmoid(dlI_pred)));
            loss_D_I_fake   = -mean(log(delta+1-sigmoid(dlI_fake_pred)));           

            loss_D_J_real   = -mean(log(delta+sigmoid(dlJ_pred)));
            loss_D_J_fake   = -mean(log(delta+1-sigmoid(dlJ_fake_pred)));

        %     loss_D_I_real   = mean((1 - sigmoid(dlI_pred)));
        %     loss_D_I_fake   = mean(sigmoid(dlI_fake_pred));           
        % 
        %     loss_D_J_real   = mean((1 - sigmoid(dlJ_pred)));
        %     loss_D_J_fake   = mean(sigmoid(dlJ_fake_pred));           

            loss_D_I        = loss_D_I_real + loss_D_I_fake;
            loss_D_J        = loss_D_J_real + loss_D_J_fake;

            loss_cyc        = (mse(dlI,dlI_recon) + mse(dlJ,dlJ_recon))/2;   % L2 like norm
        %    loss_cyc        = (mean(abs(dlI(:)-dlI_fakefake(:))) + mean(abs(dlJ(:)-dlJ_fakefake(:))))/2;    % L1 norm

            loss_encoder    = - loss_D_J_fake + gamma*loss_cyc;
            loss_decoder    = - loss_D_I_fake + gamma*loss_cyc;

            loss_cyc_enc    = gamma*mse(dlJ,dlJ_recon)/2;
            loss_cyc_dec    = gamma*mse(dlI,dlI_recon)/2;
        case 'softLabels'
            delta           = 0.4;                                                          % s.t real_lbl in [0.8 1.2] and fake_lbl in [0 0.4]                  
            soft_lbl_I_fake = 1 - (rand(size(dlI_fake_pred))*delta-delta/2);                % lbl_fake =1, lbl_real = 0;
            soft_lbl_I_real = rand(size(dlI_pred))*delta; 
            soft_lbl_J_fake = 1 - (rand(size(dlJ_fake_pred))*delta-delta/2);                 
            soft_lbl_J_real = rand(size(dlJ_pred))*delta; 

            loss_D_I_fake   = mean(log( (soft_lbl_I_fake - sigmoid(dlI_fake_pred)).^2 ));           
            loss_D_I_real   = mean(log( (soft_lbl_I_real - sigmoid(dlI_pred)).^2 )); 

            loss_D_J_fake   = mean(log( (soft_lbl_J_fake - sigmoid(dlJ_fake_pred)).^2 ));           
            loss_D_J_real   = mean(log( (soft_lbl_J_real - sigmoid(dlJ_pred)).^2 )); 

            loss_Dec_I_fake = mean(log( (soft_lbl_I_real - sigmoid(dlI_fake_pred)).^2 ));           
            loss_Enc_J_fake = mean(log( (soft_lbl_J_real - sigmoid(dlJ_fake_pred)).^2 ));           
            
            loss_D_I        = loss_D_I_real + loss_D_I_fake;
            loss_D_J        = loss_D_J_real + loss_D_J_fake;
            loss_cyc        = (mse(dlI,dlI_recon) + mse(dlJ,dlJ_recon))/2;   % L2 like norm
            loss_id         = (mse(dlI,dlI_id) + mse(dlJ,dlJ_id))/2; 
            
            loss_decoder    = loss_Enc_J_fake + loss_Dec_I_fake + gamma*loss_cyc + gamma*0.01*loss_id;            
%            loss_decoder    = loss_Enc_J_fake + loss_Dec_I_fake;
            loss_encoder    = loss_decoder;                        
        case 'softLabels-noLog'
            delta           = 0.4;                                 
            soft_1          = 1 - (rand(size(dlI_fake_pred))*delta-delta/2); 
            soft_0          = rand(size(dlI_pred))*delta; 

            loss_Dec_I_fake = mean( (sigmoid(dlI_fake_pred) - soft_1).^2 );           
            loss_Enc_J_fake = mean( (sigmoid(dlJ_fake_pred) - soft_1).^2 );           
            
            loss_D_I        = mean( (sigmoid(dlI_pred)-soft_1).^2 )  + mean( (sigmoid(dlI_fake_pred)-soft_0).^2 );
            loss_D_J        = mean( (sigmoid(dlJ_pred)-soft_1).^2 )  + mean( (sigmoid(dlJ_fake_pred)-soft_0).^2 );
            loss_cyc        = mean(abs(dlI(:)-dlI_recon(:))) + mean(abs(dlJ(:)-dlJ_recon(:)));   % L1 like norm
            loss_id         = mean(abs(dlI(:)-dlI_id(:)))    + mean(abs(dlJ(:)-dlJ_id(:)));   % L1 like norm
            
            loss_decoder    = loss_Enc_J_fake + loss_Dec_I_fake + gamma*loss_cyc + gamma*0.01*loss_id;            
%            loss_decoder    = loss_Enc_J_fake + loss_Dec_I_fake;
            loss_encoder    = loss_decoder;            

    end
end






