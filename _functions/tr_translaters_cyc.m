% Igt   - gt style images
% Iexp  - exp style images
% enc(Igt)  = Iexp_fake
%             dec(Iexp_fake) = Igt_rec
% dec(Iexp) = Igt_fake
%             enc(Igt_fake)  = I_exp_rec

function [ENC DEC D_Igt D_Iexp tr_info] = tr_translaters_cyc ...
         (ENC,DEC,D_Igt,D_Iexp,trData,pram)

  dir_valRes  = ['./_Figs/' date '/tr_translaters_cyc/'];
  mkdir(dir_valRes)
  figure; set(gcf, 'Units', 'Inches', 'Position', [1,1,18,9])
  iteration   = 0;
  start       = tic;
      
  N_iterations  = pram.numEpochs * (floor(size(trData.I_gt_tr,4)/pram.miniBatchSize)-1);
  for epch = 1:pram.numEpochs           
    b_shuffle       = randperm(size(trData.I_gt_tr,4));
    trData.I_gt_tr  = trData.I_gt_tr (:,:,:,b_shuffle);
    trData.I_exp_tr = trData.I_exp_tr(:,:,:,b_shuffle);
    
    for iter = 1:floor(size(trData.I_gt_tr,4)/pram.miniBatchSize)-1
      iteration = iteration + 1;          

      mb_str    = (iter-1)*pram.miniBatchSize+1;
      mb_end    = mb_str+pram.miniBatchSize-1;          
      Igt_tr    = gpuArray(dlarray(trData.I_gt_tr (:,:,:,mb_str:mb_end),'SSCB'));
      Iexp_tr   = gpuArray(dlarray(trData.I_exp_tr(:,:,:,mb_str:mb_end),'SSCB'));

      % Evaluate the model gradients and the generator state
      [grad_ENC,...
       grad_DEC,...
       grad_D_Igt,...
       grad_D_Iexp,...
       state_ENC,...             
       state_DEC,...             
       losses_itr] = dlfeval(@f_modelGradients,...
                              ENC, DEC, D_Igt, D_Iexp,...
                              Igt_tr, Iexp_tr,...
                              epch, iteration,pram);    
                            
      ENC.State      = state_ENC;
      DEC.State      = state_DEC;

      allLosses(iteration,:)  = extractdata(losses_itr);  % track losses
            
      % Update the D_Igt network parameters.
      [D_Igt.Learnables,pram.trailingAvg_D_I,pram.trailingAvgSq_D_I] = ...
          adamupdate(D_Igt.Learnables, grad_D_Igt, ...
                     pram.trailingAvg_D_I , pram.trailingAvgSq_D_I, iteration, ...
                     pram.learnRateDiscriminator, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);
      % Update the D_Iexp network parameters.
      [D_Iexp.Learnables,pram.trailingAvg_D_J,pram.trailingAvgSq_D_J] = ...
          adamupdate(D_Iexp.Learnables, grad_D_Iexp, ...
                     pram.trailingAvg_D_J , pram.trailingAvgSq_D_J, iteration, ...
                     pram.learnRateDiscriminator, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);                
      % Update the encoder network parameters.
      [ENC.Learnables,pram.trailingAvg_encoder,pram.trailingAvgSq_encoder] = ...
          adamupdate(ENC.Learnables, grad_ENC, ...
                     pram.trailingAvg_encoder, pram.trailingAvgSq_encoder, iteration, ...
                     pram.learnRate_encoder, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);
      % Update the encoder network parameters.
      [DEC.Learnables,pram.trailingAvg_decoder,pram.trailingAvgSq_decoder] = ...
          adamupdate(DEC.Learnables, grad_DEC, ...
                     pram.trailingAvg_decoder, pram.trailingAvgSq_decoder, iteration, ...
                     pram.learnRate_decoder, pram.gradientDecayFactor, pram.squaredGradientDecayFactor);
                    
      % Every 20 iterations, display validation results                    
      if mod(iteration,5) == 0 || iteration == 1                           
        N_val     = 4;
        b_shuf_vl = randperm(size(trData.I_gt_vl,4));
        b_inds    = b_shuf_vl(1:N_val);

        Igt_vl    = gpuArray(dlarray(trData.I_gt_vl (:,:,:,b_inds),'SSCB'));
        Iexp_vl   = gpuArray(dlarray(trData.I_exp_vl(:,:,:,b_inds),'SSCB'));

        Iexp_fake = predict(ENC,Igt_vl);
        Igt_rec   = predict(DEC,Iexp_fake);
        Igt_fake  = predict(DEC,Iexp_vl);
        Iexp_rec  = predict(ENC,Igt_fake);    
        Iexp_id   = predict(ENC,Iexp_vl);
        Igt_id    = predict(DEC,Igt_vl);
        
%         I         = imtile(cat(1, rescale(extractdata(Igt_vl)),...
%                                   rescale(extractdata(Igt_id)),...
%                                   rescale(extractdata(Iexp_fake)),...
%                                   rescale(extractdata(Igt_rec)),...
%                                   rescale(extractdata(Iexp_vl)),...
%                                   rescale(extractdata(Iexp_id)),...
%                                   rescale(extractdata(Igt_fake)),...
%                                   rescale(extractdata(Iexp_rec))),...
%                            'GridSize',[1 N_val]);

       I         = imtile(cat(2, extractdata(Igt_vl),...
                                 extractdata(Igt_id),...
                                 extractdata(Iexp_fake),...
                                 extractdata(Igt_rec),...
                                 extractdata(Iexp_vl),...
                                 extractdata(Iexp_id),...
                                 extractdata(Igt_fake),...
                                 extractdata(Iexp_rec)),...
                         'GridSize',[N_val 1]);

        subplot(2,2,1);imagesc([extractdata(Igt_vl  (:,:,1,1))...
                                extractdata(Igt_fake(:,:,1,1))...
                                extractdata(Iexp_vl (:,:,1,1))]); axis image;axis off;colorbar
        subplot(2,2,2);imagesc(I);                                axis image;axis off;colorbar
        subplot(2,2,3);semilogy(allLosses(:,1:4));                legend('L Enc','L Dec','L D I','L D J') 
        subplot(2,2,4);semilogy(allLosses(:,5:6));                legend('Lcyc','Lid')

        % Update the title with training progress information.
        t_duration      = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
            "Epoch: " + i + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(t_duration))
        drawnow
        saveas(gca,sprintf('%sitr_%0.6d.jpeg',dir_valRes,iteration));
      end      
    end
    % update learning rates
    if mod(epch,pram.learnRateDropInterval) == 0
      pram.learnRateDiscriminator = pram.learnRateDiscriminator * pram.learnRateDropFactor;
      pram.learnRate_encoder      = pram.learnRate_encoder      * pram.learnRateDropFactor;
      pram.learnRate_decoder      = pram.learnRate_decoder      * pram.learnRateDropFactor;
    end
  end    
  allLosses           = gather(allLosses);  
  tr_info.loss_encoder   = allLosses(:,1);
  tr_info.loss_decoder   = allLosses(:,2);
  tr_info.loss_D_I       = allLosses(:,3);
  tr_info.loss_D_J       = allLosses(:,4);    
end


%% model gradient function
function    [grad_ENC,...
             grad_DEC,...
             grad_D_Igt,...
             grad_D_Iexp,...
             state_ENC,...             
             state_DEC,...             
             losses_itr] = f_modelGradients(ENC, DEC, D_Igt, D_Iexp,...
                                            Igt, Iexp,...
                                            epoch, iteration, pram)

  % calculate translations and predictions from the discriminators
  [Iexp_fake  state_ENC]  = forward(ENC,Igt);
  [Igt_rec    state_DEC]  = forward(DEC,Iexp_fake);
  [Igt_fake   state_DEC]  = forward(DEC,Iexp);
  [Iexp_rec   state_ENC]  = forward(ENC,Igt_fake);    
  [Iexp_id    state_ENC]  = forward(ENC,Iexp);
  [Igt_id     state_DEC]  = forward(DEC,Igt);
  Igt_pred                = forward(D_Igt, Igt);
  Iexp_pred               = forward(D_Iexp, Iexp);
  Igt_fake_pred           = forward(D_Igt, Igt_fake);
  Iexp_fake_pred          = forward(D_Iexp, Iexp_fake);

  % mse's for cyclic and id losses    
  loss_cyc            = (mse(Igt,Igt_rec) + mse(Iexp,Iexp_rec))/2;   % L2 like norm  
  loss_id             = (mse(Igt,Igt_id ) + mse(Iexp,Iexp_id ))/2; 

  % Convert the discriminator outputs to probabilities.
  prob_Igt_pred       = sigmoid(Igt_pred);
  prob_Iexp_pred      = sigmoid(Iexp_pred);
  prob_Igt_fake_pred  = sigmoid(Igt_fake_pred);
  prob_Iexp_fake_pred = sigmoid(Iexp_fake_pred);

  % Randomly flip a fraction of the labels of the real images.
  flipFactor = 0.1;
  numObservations = size(prob_Igt_pred,4);
  idx = randperm(numObservations,floor(flipFactor * numObservations));
  prob_Igt_pred(:,:,:,idx)  = 1-prob_Igt_pred(:,:,:,idx);                          % Flip the labels    
  prob_Iexp_pred(:,:,:,idx) = 1-prob_Iexp_pred(:,:,:,idx);                         % Flip the labels    
  
  % Calculate the GAN loss
  [lossAd_ENC, lossAd_DEC, loss_D_Igt, loss_D_Iexp] = f_ganLoss(prob_Igt_pred,...
                                                                prob_Iexp_pred,...
                                                                prob_Igt_fake_pred,...
                                                                prob_Iexp_fake_pred);
                                                               
  loss_ENC = lossAd_ENC + pram.gammaCyc*loss_cyc + pram.gammaId*loss_id;   
  loss_DEC = lossAd_DEC + pram.gammaCyc*loss_cyc + pram.gammaId*loss_id;   

  losses_itr = [loss_ENC, loss_DEC, loss_D_Igt, loss_D_Iexp, loss_cyc, loss_id];

  % For each network, calculate the gradients with respect to the loss.
  grad_ENC    = dlgradient(loss_ENC,    ENC.Learnables,   'RetainData',true);    
  grad_DEC    = dlgradient(loss_DEC,    DEC.Learnables,   'RetainData',true);    
  grad_D_Igt  = dlgradient(loss_D_Igt,  D_Igt.Learnables);
  grad_D_Iexp = dlgradient(loss_D_Iexp, D_Iexp.Learnables);
  
  fprintf('|ep=%d | itr=%d || L_ENC=%d | L_DEC=%d | L_cyc=%d | L_id=%d | L_DIgt=%d | L_DIexp=%d | (g==0?)=[%d %d %d %d] \n',...
           epoch,   iteration,loss_ENC,  loss_ENC,  loss_cyc,  loss_id,  loss_D_Igt, loss_D_Iexp,    squeeze(extractdata(grad_ENC.Value{end}(1)))==0,...
                                                                                                        squeeze(extractdata(grad_DEC.Value{end}(1)))==0,...
                                                                                                           squeeze(extractdata(grad_D_Igt.Value{end}(1)))==0,...
                                                                                                              squeeze(extractdata(grad_D_Iexp.Value{end}(1)))==0 );    

end


function [loss_ENC, loss_DEC, loss_D_Igt, loss_D_Iexp] = f_ganLoss(prob_Igt_pred,...
                                                                   prob_Iexp_pred,...
                                                                   prob_Igt_fake_pred,...
                                                                   prob_Iexp_fake_pred)
  delta       = 1e-20;
  loss_D_Igt  = -mean(log(prob_Igt_pred +delta)) -mean(log(1-prob_Igt_fake_pred +delta));
  loss_D_Iexp = -mean(log(prob_Iexp_pred+delta)) -mean(log(1-prob_Iexp_fake_pred+delta));

  loss_ENC    = -mean(log(prob_Iexp_fake_pred+delta));    
  loss_DEC    = -mean(log(prob_Igt_fake_pred +delta));                                                                
end





