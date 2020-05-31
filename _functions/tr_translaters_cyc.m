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
        imds_I_tr = shuffle(imds_I_tr);
        imds_J_tr = shuffle(imds_J_tr);
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
                dlJ_fake    = predict(encoder,dlI_val);
                dlI_fake    = predict(decoder,dlJ_val);
         
%                 img_I       = rescale(imtile(cat(2, extractdata(dlI_val),extractdata(dlI_fake))));
%                 img_J       = rescale(imtile(cat(2, extractdata(dlJ_val),extractdata(dlJ_fake))));                
                img_I       = imtile(cat(2, extractdata(dlI_val),extractdata(dlI_fake)));
                img_J       = imtile(cat(2, extractdata(dlJ_val),extractdata(dlJ_fake)));

                imagesc([img_I img_J]);axis image

                
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
    [dlJ_fake       state_encoder]  = forward(encoder,dlI);
    [dlI_fake       state_decoder]  = forward(decoder,dlJ);
    [dlJ_fakefake   state_encoder]  = forward(encoder,dlI_fake);
    [dlI_fakefake   state_decoder]  = forward(decoder,dlJ_fake);
    
    dlI_pred            = forward(D_I, dlI);
    dlJ_pred            = forward(D_J, dlJ);
    dlI_fake_pred       = forward(D_I, dlI_fake);
    dlJ_fake_pred       = forward(D_J, dlJ_fake);
    dlI_fakefake_pred   = forward(D_I, dlI_fakefake);    
    dlJ_fakefake_pred   = forward(D_J, dlJ_fakefake);    

    % Calculate the GAN loss
    [loss_encoder, loss_decoder, loss_D_I, loss_D_J, loss_cyc_enc, loss_cyc_dec] = ...
                                                                 f_ganLoss(dlI_pred,...
                                                                 dlJ_pred,...
                                                                 dlI_fake_pred,...
                                                                 dlJ_fake_pred,...
                                                                 dlI,dlI_fakefake,...
                                                                 dlJ,dlJ_fakefake,...
                                                                 pram.gammaCyc);

    disp(sprintf('%d %d\t L_encCyc %d\t L_decCyc %d  \t L_enc %d \t L_dec %d \t L_DI %d \t L_DJ %d',...
                epoch,iteration, loss_cyc_enc, loss_cyc_dec, loss_encoder, loss_decoder, loss_D_I, loss_D_J));    
    
    losses_itr = [loss_encoder, loss_decoder, loss_D_I, loss_D_J, loss_cyc_enc, loss_cyc_dec];

    % For each network, calculate the gradients with respect to the loss.
    gradients_encoder   = dlgradient(loss_encoder, encoder.Learnables,'RetainData',true);    
    gradients_decoder   = dlgradient(loss_decoder, decoder.Learnables,'RetainData',true);    
    gradients_D_I       = dlgradient(loss_D_I    , D_I.Learnables);
    gradients_D_J       = dlgradient(loss_D_J    , D_J.Learnables);
end


function [loss_encoder, loss_decoder, loss_D_I, loss_D_J, loss_cyc_enc, loss_cyc_dec] = ...
                                                                      f_ganLoss(dlI_pred,...
                                                                      dlJ_pred,...
                                                                      dlI_fake_pred,...
                                                                      dlJ_fake_pred,...
                                                                      dlI,dlI_fakefake,...
                                                                      dlJ,dlJ_fakefake,...
                                                                      gamma)
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

    loss_cyc        = (mse(dlI,dlI_fakefake) + mse(dlJ,dlJ_fakefake))/2;   % L2 like norm
%    loss_cyc        = (mean(abs(dlI(:)-dlI_fakefake(:))) + mean(abs(dlJ(:)-dlJ_fakefake(:))))/2;    % L1 norm
    
    loss_encoder    = - loss_D_J_fake + gamma*loss_cyc;
    loss_decoder    = - loss_D_I_fake + gamma*loss_cyc;

    loss_cyc_enc    = gamma*mse(dlJ,dlJ_fakefake)/2;
    loss_cyc_dec    = gamma*mse(dlI,dlI_fakefake)/2;
end






