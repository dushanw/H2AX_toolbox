
function [imgprocessor discriminator info] = tr_segmenter_adv(imgprocessor,discriminator,trData,pram)
  figure; set(gcf, 'Units', 'Inches', 'Position', [1,1,18,9])
  iteration = 0;
  start     = tic;
    
  N_iterations  = pram.numEpochs * (floor(size(trData.I_tr,4)/pram.miniBatchSize)-1);
  for epch = 1:pram.numEpochs 
    b_shuffle   = randperm(size(trData.I_tr,4));
    trData.I_tr = trData.I_tr(:,:,:,b_shuffle);
    trData.L_tr = trData.L_tr(:,:,:,b_shuffle);
    
    for iter = 1:floor(size(trData.I_tr,4)/pram.miniBatchSize)-1
          iteration = iteration + 1;          
          
          mb_str    = (iter-1)*pram.miniBatchSize+1;
          mb_end    = mb_str+pram.miniBatchSize-1;          
          I_tr      = gpuArray(dlarray(trData.I_tr(:,:,:,mb_str:mb_end),'SSCB'));
          L_tr      = gpuArray(dlarray(trData.L_tr(:,:,:,mb_str:mb_end),'SSCB'));          
          L_tr      = cat(3,cat(3,L_tr(:,:,1,:)*-1+1,L_tr(:,:,1:2,:)));       % ch-dim is [bg, fg hand-annotated-bg]

          % Evaluate the model gradients and the generator state
          [grad_G,...
           grad_D,... 
           state_imgprocessor,...             
           losses_itr] = dlfeval(@f_modelGradients,...
                                  imgprocessor, discriminator,...
                                  I_tr, L_tr, L_tr_wbg,...
                                  pram, iter, iteration);
          imgprocessor.State      = state_imgprocessor;  
          
          % track and display iteration details 
          allLosses(iteration,:)  = extractdata(losses_itr);  % track losses
          disp(sprintf('epoch = %d(of %d) | iter=%d(of %d) || L_G=%d | L_D=%d | L_mse=%d | grad_G=%d | grad_D=%d',...
                                epch,pram.numEpochs,...
                                            iteration,N_iterations,...         
                                                              allLosses(iteration,1),...
                                                                         allLosses(iteration,2),...
                                                                                  allLosses(iteration,3),...
                                                                                             squeeze(extractdata(grad_G.Value{end}(1))),...
                                                                                                         squeeze(extractdata(grad_D.Value{end}(1))) )); 

          

          % Update the discriminator network parameters.
          [discriminator.Learnables,pram.trailingAvgDiscriminator,pram.trailingAvgSqDiscriminator] = ...
              adamupdate(discriminator.Learnables, grad_D, ...
                         pram.trailingAvgDiscriminator , pram.trailingAvgSqDiscriminator, iteration, ...
                         pram.learnRateDiscriminator   , pram.gradientDecayFactor,        pram.squaredGradientDecayFactor);

           % Update the imageprocesser network parameters.
          [imgprocessor.Learnables,pram.trailingAvgImgprocessor,pram.trailingAvgSqImgprocessor] = ...
              adamupdate(imgprocessor.Learnables, grad_G, ...
                         pram.trailingAvgImgprocessor , pram.trailingAvgSqImgprocessor, iteration, ...
                         pram.learnRateImgprocessor   , pram.gradientDecayFactor , pram.squaredGradientDecayFactor);

          % Every 10 iterations, display validation results                    
          if mod(iteration,10) == 0 || iteration == 1
              N_val     = 4;
              b_shuf_vl = randperm(size(trData.I_vl,4));
              b_inds    = b_shuf_vl(1:N_val);
              
              I_vl      = gpuArray(dlarray(trData.I_vl(:,:,:,b_inds),'SSCB'));
              L_vl_real = gpuArray(dlarray(trData.L_vl(:,:,:,b_inds),'SSCB'));

              L_vl_fake = predict(imgprocessor,I_vl);
              I         = imtile(cat(2, extractdata(I_vl),...
                                        extractdata(L_vl_fake(:,:,2,:)+5*L_vl_fake(:,:,3,:)),... 
                                        extractdata(L_vl_real(:,:,1,:)+5*L_vl_real(:,:,2,:)))...
                                 ,'GridSize',[N_val 1]);

              subplot(1,2,1);imagesc(I,[0 5]);axis image;colorbar
              subplot(1,2,2);plot(allLosses);
                             legend('LG','LD','Lmse') 

              % Update the title with training progress information.
              D = duration(0,0,toc(start),'Format','hh:mm:ss');
              title(...
                  "Epoch: " + epch + ", " + ...
                  "Iteration: " + iteration + ", " + ...
                  "Elapsed: " + string(D))

              drawnow
          end            
      end
  end

  allLosses               = gather(allLosses);  
  info.loss_imgprocessor  = allLosses(:,1);
  info.loss_discriminator = allLosses(:,2);
  info.loss_mseLike       = allLosses(:,3);
end


%% model gradient function                                    
function [grad_G, grad_D, state_G, allLosses] = f_modelGradients(G, D, I, L_real,L_real_w, pram,epoch,iteration)
  
  % Calculate the predictions for real data with the discriminator network.
  YPred = forward(D, L_real);

  % Calculate the predictions for generated data with the discriminator network.
  [L_fake,state_G] = forward(G,I);
  YPred_fake = forward(D,L_fake);% second channel is the 'fg'

  cross_entrpy    = crossentropy(L_fake(:,:,1:2,:),L_real(:,:,1:2,:))/(pram.Nx^2);
  w_cross_entrpy  = crossentropy(L_fake(:,:,3,:),L_real(:,:,3,:))/(pram.Nx^2);  
  mse             = (cross_entrpy + 10*w_cross_entrpy);
%  mse = mean((L_fake(:) - L_real(:)).^2);
  
  % Convert the discriminator outputs to probabilities.
  probGenerated = sigmoid(YPred_fake);
  probReal = sigmoid(YPred);
  % dlY = crossentropy(dlXGenerated,dlL);


  % Calculate the score of the discriminator.
  scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

  % Calculate the score of the generator.
  scoreGenerator = mean(probGenerated);
  
  % Randomly flip a fraction of the labels of the real images.
  flipFactor = 0.1;
  numObservations = size(probReal,4);
  idx = randperm(numObservations,floor(flipFactor * numObservations));

  % Flip the labels
  probReal(:,:,:,idx) = 1-probReal(:,:,:,idx);

  % Calculate the GAN loss.
  [advlossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated);
  lossGenerator = advlossGenerator + pram.gammaMse*mse;
  
  % For each network, calculate the gradients with respect to the loss.
  grad_G = dlgradient(lossGenerator, G.Learnables,'RetainData',true);
  grad_D = dlgradient(lossDiscriminator, D.Learnables);

  allLosses = [advlossGenerator, lossDiscriminator, pram.gammaMse*mse];
end

function [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated)
  % Calculate the loss for the discriminator network.
  delta = 1e-20;
  lossDiscriminator =  -mean(log(probReal+delta)) -mean(log(1-probGenerated+delta));

  % Calculate the loss for the generator network.
  lossGenerator = -mean(log(probGenerated+delta));
end










