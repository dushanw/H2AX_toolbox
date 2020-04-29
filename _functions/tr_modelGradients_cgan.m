% Model Gradients Function

function [gradientsEncoder, gradientsGenerator, gradientsDiscriminator, stateEncoder, stateGenerator, allLosses] = ...
    modelGradients_cgan(dlnetEncoder, dlnetGenerator, dlnetDiscriminator, dlX,gamma,epoch,iteration)

    % Calculate the predictions for generated data with the discriminator network.
    [dlZ,stateEncoder]              = forward(dlnetEncoder,dlX);
    [dlXGenerated,stateGenerator]   = forward(dlnetGenerator,dlZ);
    dlYPredGenerated                = forward(dlnetDiscriminator, cat(3,dlX,dlXGenerated));
    
    % Calculate the predictions for real data with the discriminator network.
    dlYPred = forward(dlnetDiscriminator, cat(3,dlX,dlX));

    
    % Calculate the GAN loss
    [lossGenerator, lossDiscriminator, allLosses] = f_ganLoss(dlYPred,dlYPredGenerated,dlX,dlXGenerated,gamma);

    disp(sprintf('%d-%d:\tenc&gen.loss = %d\tdis.loss = %d\tMSE=%d\tLD_gen=%d\t',epoch,iteration,lossGenerator,lossDiscriminator,allLosses(end),allLosses(end-1)));    
    
    % For each network, calculate the gradients with respect to the loss.
    gradientsEncoder        = dlgradient(lossGenerator, dlnetEncoder.Learnables,'RetainData',true);
    gradientsGenerator      = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
    gradientsDiscriminator  = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end

