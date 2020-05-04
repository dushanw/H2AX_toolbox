
function plt_trainingLosses(tr_info)
    FS = 14;
    
    figure;
    subplot(1,3,1)
    plot(tr_info.loss_imgprocessor);hold on
    plot(tr_info.loss_D_seg);
    set(gca,'fontsize',FS)
    xlabel('Iteration')
    ylabel('Loss')
    legend('IP','D')
    
    subplot(1,3,2)
    plot(tr_info.loss_discriminator_real);hold on
    plot(tr_info.loss_discriminator_fake);
    set(gca,'fontsize',FS)
    xlabel('Iteration')
    ylabel('Loss')
    legend('L_D(Real)','L_D(Fake)')
    
    subplot(1,3,3)
    plot(tr_info.loss_imgprocessor_adv);hold on
    plot(tr_info.loss_imgprocessor_mse);
    set(gca,'fontsize',FS)
    xlabel('Iteration')
    ylabel('Loss')
    legend('IP_{adv}','IP_{mse}')

    set(gcf,'Units','Inches','Position',[1,1,14,3])
    saveas(gca,['./__results/' date '_.png']);
end