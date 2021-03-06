function ipmi_problem_figure_quant_log

    [z_b6k_acc_vec,z_b3k_b6k_acc_vec,z_b3k_b6k_b9k_acc_vec,z_b3k_b6k_b9k_b12k_acc_vec] = ipmi_2019_zeta_opt_eval_acc;
    [b6k_acc_vec,b3k_b6k_acc_vec,b3k_b6k_b9k_acc_vec,b3k_b6k_b9k_b12k_acc_vec] = ipmi_2019_zeta_opt_eval_acc_log_v2;
    
    bins = linspace(-1,1,200);
    [zh1,zp1] = hist(z_b6k_acc_vec,bins);
    [h1,p1] = hist(b6k_acc_vec,bins);
    
    [zh2,zp2] = hist(z_b3k_b6k_acc_vec,bins);
    [h2,p2] = hist(b3k_b6k_acc_vec,bins);

    [zh3,zp3] = hist(z_b3k_b6k_b9k_acc_vec,bins);
    [h3,p3] = hist(b3k_b6k_b9k_acc_vec,bins);
    
    [zh4,zp4] = hist(z_b3k_b6k_b9k_b12k_acc_vec,bins);
    [h4,p4] = hist(b3k_b6k_b9k_b12k_acc_vec,bins);
    
    figure(4)
    subplot(2,2,1)
    plot(zh1,'r--','LineWidth',2)
    hold on
    plot(h1,'g-.','LineWidth',2)
    legend('Optimized','Log-Optimized','Location','best')
    grid on
    legend boxoff
    ylabel('No. of Voxels')
    xlabel('Angular Correlation Coefficient')
    title('B6000')    
    
    subplot(2,2,2)
    plot(zh2,'r--','LineWidth',2)
    hold on
    plot(h2,'g-.','LineWidth',2)
    legend('Optimized','Log-Optimized','Location','best')
    grid on
    legend boxoff
    ylabel('No. of Voxels')
    xlabel('Angular Correlation Coefficient')
    title('B3000, B6000')
    
    subplot(2,2,3)
    plot(zh3,'r--','LineWidth',2)
    hold on
    plot(h3,'g-.','LineWidth',2)
    legend('Optimized','Log-Optimized','Location','best')
    grid on
    legend boxoff
    ylabel('No. of Voxels')
    xlabel('Angular Correlation Coefficient')
    title('B3000, B6000, B9000')
    
    subplot(2,2,4)
    plot(zh4,'r--','LineWidth',2)
    hold on
    plot(h4,'g-.','LineWidth',2)
    legend('Optimized','Log-Optimized','Location','best')
    grid on
    legend boxoff
    ylabel('No. of Voxels')
    xlabel('Angular Correlation Coefficient')
    title('B3000, B6000, B9000, B12000')
    
end