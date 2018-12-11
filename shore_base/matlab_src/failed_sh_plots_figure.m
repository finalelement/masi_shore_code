function failed_sh_plots_figure

    % For Failed SH Plots Here is the indexing
    % 1 -  'shore_b3k_sh_6th'
    % 2 - 'shore_b3k_sh_8th'
    % 3 - 'shore_b3k_shore_fod'
    % 4 - 'shore_b9k_sh_6th'
    % 5 - 'shore_b9k_sh_8th'
    % 6 - 'shore_b9k_shore_fod'

    [all_train,all_valid] = failed_sh_plots_ipmi_2019;
    [all_train_acc,all_valid_acc] = failed_sh_plots_ipmi_2019_acc;
    
    
    figure
    subplot(1,3,1)
    plot(all_train{1},'LineWidth',1.5)
    hold on
    plot(all_valid{1},'LineWidth',1.5)
    plot(all_train{2},'LineWidth',1.5)
    plot(all_valid{2},'LineWidth',1.5)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    xlabel('Epochs/Iterations')
    ylabel('Mean Squared Error')
    legend('Training SHORE->SH 6th','Training SHORE->SH 8th','Validation SHORE->SH 6th','Validation SHORE->SH 8th','Location','Best')
    grid on
    
    
    subplot(1,3,2)
    plot(all_train_acc{1},'LineWidth',1.5)
    hold on
    plot(all_valid_acc{1},'LineWidth',1.5)
    plot(all_train_acc{2},'LineWidth',1.5)
    plot(all_valid_acc{2},'LineWidth',1.5)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    xlabel('Epochs/Iterations')
    ylabel('Angular Correlation Coefficient')
    legend('Training SHORE->SH 6th','Training SHORE->SH 8th','Validation SHORE->SH 6th','Validation SHORE->SH 8th','Location','Best')
    grid on
    
    subplot(1,3,3)
    plot(all_train{3},'LineWidth',1.5)
    hold on
    plot(all_valid{3},'LineWidth',1.5)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    xlabel('Epochs/Iterations')
    ylabel('Mean Squared Error')
    legend('Training SHORE->SHORE FOD','Validation SHORE->SHORE FOD','Location','Best')
    grid on