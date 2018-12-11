function [all_train,all_valid] = failed_sh_plots_ipmi_2019

    base_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ipmi_results\failed_results_loop50';
    
    base_path_dir_list = dir(base_path);
    iters = 100;
    
    all_train = {};
    all_valid = {};
    
    exp_counter = 1;
    
    for i = 3:length(base_path_dir_list)
    
        t_path = fullfile(base_path,base_path_dir_list(i).name);
        t_path_list = dir(t_path);
        disp(base_path_dir_list(i).name)
        kfold_counter = 1;
        
        train_d = zeros(iters,5);
        valid_d = zeros(iters,5);
        
        for j = 3:length(t_path_list)-1
            
            disp(kfold_counter)
            file_tb_read = fullfile(t_path,t_path_list(j).name,'results.csv');
            disp(file_tb_read)
            file_data = csvread(file_tb_read,1,1);
            train_d(:,kfold_counter) = file_data(:,3);
            valid_d(:,kfold_counter) = file_data(:,6);
            
            kfold_counter = kfold_counter + 1;
        end
        
        mean_train_d = mean(train_d,2);
        mean_valid_d = mean(valid_d,2);
        
        all_train{exp_counter} = mean_train_d;
        all_valid{exp_counter} = mean_valid_d;
    
        exp_counter = exp_counter + 1;
        
    end
    
    figure
    subplot(2,2,1)
    plot(all_train{1},'LineWidth',1.5)
    hold on
    plot(all_valid{1},'LineWidth',1.5)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    
    subplot(2,2,2)
    plot(all_train{2},'LineWidth',1.5)
    hold on
    plot(all_valid{2},'LineWidth',1.5)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    
    subplot(2,2,3)
    plot(all_train{3},'LineWidth',1.5)
    hold on
    plot(all_valid{3},'LineWidth',1.5)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    
    subplot(2,2,4)
    plot(all_train{6},'LineWidth',1.5)
    hold on
    plot(all_valid{6},'LineWidth',1.5)
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    
end