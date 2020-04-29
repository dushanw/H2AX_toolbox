% GAN Loss Function

function [lossGenerator, lossDiscriminator, allLosses] = ganLoss(dlYPred,dlYPredGenerated,dlX,dlXGenerated,gamma)

    delta = 1e-3;
    % Calculate losses for the discriminator network.
    d_lossGenerated     = -mean(log(delta+1-sigmoid(dlYPredGenerated)));
    d_lossReal          = -mean(log(delta+sigmoid(dlYPred)));
    % Calculate losses for the generator network.
    g_lossGenerated     = -mean(log(delta+sigmoid(dlYPredGenerated)));
    lossMSE             = mse(dlX,dlXGenerated);
    
    if isnan(d_lossGenerated.extractdata) | isinf(d_lossGenerated.extractdata)
       xx=1 
    end

    % Combine the losses for the discriminator network.
    lossDiscriminator   = d_lossReal + d_lossGenerated;

    % Calculate the loss for the generator network.
%    lossGenerator       = lossMSE*gamma;
%    lossGenerator       = g_lossGenerated/gamma + lossMSE;
    lossGenerator       = g_lossGenerated + lossMSE*gamma;
%    lossGenerator       = g_lossGenerated;

    allLosses = [lossGenerator lossDiscriminator d_lossGenerated d_lossReal g_lossGenerated lossMSE];
end