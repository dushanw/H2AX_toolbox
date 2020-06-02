

function [imgprocessor discriminator info] = tr_segmenter_adv(imgprocessor,discriminator,imds_gt,pxds_gt,pram)

    %% network parameters 
    patchSize       = [pram.Nx pram.Nx];
    miniBatchSize   = pram.miniBatchSize;

    %% set other datastore parameters
    I_temp          = imds_gt.read;     
    imds_gt.reset;    
    N_PatchesPerImg = round(prod(size(I_temp)./patchSize)*0.05);
    
    %% image-patch datastores
    patchds_tr                  = randomPatchExtractionDatastore(imds_gt,pxds_gt,patchSize,'PatchesPerImage',N_PatchesPerImg);
    patchds_tr.MiniBatchSize    = miniBatchSize;

    figure
    iteration       = 0;
    start           = tic;
    
    for i = 1:pram.numEpochs

        % Reset and shuffle datastore.
        reset(patchds_tr);
        patchds_tr = shuffle(patchds_tr);
        
        data_val        = read(patchds_tr);
        I_val           = cat(4,data_val.InputImage{:});
        L_val(:,:,1,:)  = single(cat(4,data_val.ResponsePixelLabelImage{:})==pram.classNames(1));
        L_val(:,:,2,:)  = single(cat(4,data_val.ResponsePixelLabelImage{:})==pram.classNames(2));

        dlI_val         = dlarray(I_val, 'SSCB');
        dlL_val         = dlarray(L_val, 'SSCB');

        if (pram.executionEnvironment == "auto" && canUseGPU) || pram.executionEnvironment == "gpu"
            dlI_val     = gpuArray(dlI_val);
            dlL_val     = gpuArray(dlL_val);
        end



        while hasdata(patchds_tr)
            
            data_tr         = read(patchds_tr);
            if size(data_tr,1)<pram.miniBatchSize
                break
            end
            iteration   = iteration + 1;
            
            I_tr            = cat(4,data_tr.InputImage{:});
            L_tr(:,:,1,:)   = single(cat(4,data_tr.ResponsePixelLabelImage{:})==pram.classNames(1));
            L_tr(:,:,2,:)   = single(cat(4,data_tr.ResponsePixelLabelImage{:})==pram.classNames(2));

            dlI_tr          = dlarray(I_tr, 'SSCB');
            dlL_tr          = dlarray(L_tr, 'SSCB');
            
            if (pram.executionEnvironment == "auto" && canUseGPU) || pram.executionEnvironment == "gpu"
                dlI_tr      = gpuArray(dlI_tr);
                dlL_tr      = gpuArray(dlL_tr);
            end
            
            % Evaluate the model gradients and the generator state
            [gradients_imgprocessor,...
             gradients_discriminator,... 
             state_imgprocessor,...             
             losses_itr] = dlfeval(@f_modelGradients,...
                                    imgprocessor, discriminator,...
                                    dlI_tr, dlL_tr,...
                                    pram.gammaMse, i, iteration);

            imgprocessor.State      = state_imgprocessor;            

            allLosses(iteration,:)  = extractdata(losses_itr);  % track losses
            
            % Update the discriminator network parameters.
            [discriminator.Learnables,pram.trailingAvgDiscriminator,pram.trailingAvgSqDiscriminator] = ...
                adamupdate(discriminator.Learnables, gradients_discriminator, ...
                           pram.trailingAvgDiscriminator , pram.trailingAvgSqDiscriminator, iteration, ...
                           pram.learnRateDiscriminator   , pram.gradientDecayFactor,        pram.squaredGradientDecayFactor);
                
             % Update the imageprocesser network parameters.
            [imgprocessor.Learnables,pram.trailingAvgImgprocessor,pram.trailingAvgSqImgprocessor] = ...
                adamupdate(imgprocessor.Learnables, gradients_imgprocessor, ...
                           pram.trailingAvgImgprocessor , pram.trailingAvgSqImgprocessor, iteration, ...
                           pram.learnRateImgprocessor   , pram.gradientDecayFactor , pram.squaredGradientDecayFactor);

                    
            % Every 100 iterations, display validation results                    
            if mod(iteration,20) == 0 || iteration == 1            
                dlL_val_gen    = predict(imgprocessor,dlI_val);
         
%                 I   = imtile(cat(2,extractdata(dlI_val),extractdata(dlL_val_gen(:,:,2,:)),extractdata(dlL_val(:,:,2,:))));
%                 I   = rescale(I);
%   
                I   = imtile(cat(2,extractdata(dlI_val(:,:,:,1:6)),extractdata(dlL_val_gen(:,:,2,1:6)),extractdata(dlL_val(:,:,2,1:6)))...
                             ,'GridSize',[6 1]);
%                I   = rescale(I);
  
                
                subplot(1,3,1);imagesc(I);axis image;colorbar
                subplot(1,3,2);plot(allLosses(:,1:end-1));
                               legend('LG','LD','L D(G(z))','L D(x)','LG adv') 
                subplot(1,3,3);plot(allLosses(:,end));
                               legend('L mse') 

                               
                % Update the title with training progress information.
                D = duration(0,0,toc(start),'Format','hh:mm:ss');
                title(...
                    "Epoch: " + i + ", " + ...
                    "Iteration: " + iteration + ", " + ...
                    "Elapsed: " + string(D))

                drawnow
            end            
        end
    end
    
    allLosses                       = gather(allLosses);  
    info.loss_imgprocessor          = allLosses(:,1);
    info.loss_discriminator         = allLosses(:,2);
    info.loss_discriminator_fake    = allLosses(:,3);
    info.loss_discriminator_real    = allLosses(:,4);
    info.loss_imgprocessor_adv      = allLosses(:,5);
    info.loss_imgprocessor_mse      = allLosses(:,6);
end


%% model gradient function
function [gradients_imgprocessor,...
          gradients_discriminator,... 
          state_imgprocessor,...             
          allLosses] = f_modelGradients(imgprocessor, discriminator,...
                                     dlI, dlL,...
                                     gamma,epoch,iteration)

    % Calculate the predictions for generated data with the discriminator network.
    [dlL_gen,state_imgprocessor]    = forward(imgprocessor,dlI);
    dlYPred_gen                     = forward(discriminator, cat(3,dlI,dlL_gen));
    
    % Calculate the predictions for real data with the discriminator network.
    dlYPred                         = forward(discriminator, cat(3,dlI,dlL));

    % Calculate the GAN loss
    [loss_imgprocessor, loss_discriminator, allLosses] = f_ganLoss(dlYPred,dlYPred_gen,dlL,dlL_gen,gamma);

%    disp(sprintf('%d-%d:\timgProcLoss.loss = %d\tdis.loss = %d\tMSE=%d\tLD_gen=%d\t',epoch,iteration,loss_imgprocessor,loss_discriminator,allLosses(end),allLosses(end-1)));    
    disp(sprintf('%d %d\t LG %d \t LD %d \t Lmse %d',epoch,iteration,loss_imgprocessor,loss_discriminator,allLosses(end)));    
    

    % For each network, calculate the gradients with respect to the loss.
    gradients_imgprocessor          = dlgradient(loss_imgprocessor , imgprocessor.Learnables,'RetainData',true);    
    gradients_discriminator         = dlgradient(loss_discriminator, discriminator.Learnables);
end


function [loss_generator, loss_discriminator, allLosses] = f_ganLoss(dlYPred_real,dlYPred_fake,dlL,dlL_gen,gamma)
    
    loss_mode = 'softLabels';
    switch loss_mode
        case 'slack'
            delta = 1e-6;                                                               % slack value

            d_loss_fake     = -mean(log(delta+1-sigmoid(dlYPred_fake)));                 % calculate losses for the discriminator network.
            d_loss_real     = -mean(log(delta+sigmoid(dlYPred_real)));

            g_loss_fake      = -mean(log(delta+sigmoid(dlYPred_fake)));                   % calculate losses for the generator network.
            loss_mse        = mse(dlL,dlL_gen);

            if isnan(d_loss_fake.extractdata) | isinf(d_loss_fake.extractdata)
               xx=1 
            end

            loss_discriminator   = d_loss_real + d_loss_fake;                           % combine the losses for the discriminator network.  
            loss_generator       = g_loss_fake + loss_mse*gamma;                         % calculate the loss for the generator network.   

            allLosses = [loss_generator loss_discriminator d_loss_fake d_loss_real g_loss_fake loss_mse*gamma];
        case 'WGAN'        
            d_fake              = mean(dlYPred_fake);                                    % calculate losses for the discriminator network.
            d_real              = mean(dlYPred_real);
            loss_mse            = mse(dlL,dlL_gen);

            loss_discriminator  = - (d_real - d_fake);
            loss_generator      = - d_fake + loss_mse*gamma;                            % calculate the loss for the generator network.   

            allLosses = [loss_generator loss_discriminator d_fake d_real 0 loss_mse];
        case 'cap'
            cap_value       = 1e-10; 
            d_loss_fake     = -mean(log(max(1-sigmoid(dlYPred_fake),cap_value)));        % calculate losses for the discriminator network.
            d_loss_real     = -mean(log(max(sigmoid(dlYPred_real),2*cap_value)));

            g_loss_fake      = -mean(log(max(sigmoid(dlYPred_fake),cap_value)));          % calculate losses for the generator network.
            loss_mse        = mse(dlL,dlL_gen);
            
            loss_discriminator   = d_loss_real + d_loss_fake;                           % combine the losses for the discriminator network.  
            loss_generator       = g_loss_fake + loss_mse*gamma;                         % calculate the loss for the generator network.   

            allLosses = [loss_generator loss_discriminator d_loss_fake d_loss_real g_loss_fake loss_mse];
        case 'softLabels'
            delta           = 0.4;                                                          % s.t real_lbl in [0.8 1.2] and fake_lbl in [0 0.4]                  
            soft_lbl_fake   = 1 - (rand(size(dlYPred_fake))*delta-delta/2);                 % lbl_fake =1, lbl_real = 0;
            soft_lbl_real   = rand(size(dlYPred_fake))*delta; 
            
            d_loss_fake     = mean(log( (soft_lbl_fake - sigmoid(dlYPred_fake)).^2 ));     % calculate losses for the discriminator network.
            d_loss_real     = mean(log( (soft_lbl_real - sigmoid(dlYPred_real)).^2 ));

            g_loss_fake     = mean(log( (soft_lbl_real - sigmoid(dlYPred_real)).^2 ));     % fliped labels
            loss_mse        = mse(dlL,dlL_gen);
            
            loss_discriminator   = d_loss_real + d_loss_fake;                               % combine the losses for the discriminator network.  
            loss_generator       = g_loss_fake + loss_mse*gamma;                            % calculate the loss for the generator network.   

            allLosses = [loss_generator loss_discriminator d_loss_fake d_loss_real g_loss_fake loss_mse];

    end
end











