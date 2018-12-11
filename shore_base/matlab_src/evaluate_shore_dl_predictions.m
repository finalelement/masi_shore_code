function evaluate_shore_dl_predictions

    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Shore_4_shells_trial_1\testing_results\test_results.mat')
    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Shore_4_shells_trial_3_r8\Hist_Blind_72_Test\result.mat')
    
    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Mapmri_4_shells_trial_1_r6\Hist_Blind_72_Test\result.mat')
    
    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Mapmri_4_shells_trial_2_r8\Hist_Blind_72_Test\result.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Mapmri_4_shells_trial_3_r8\Hist_Blind_72_Test\result.mat')
    
    acc_vec = zeros(length(out_true),1);
    
    for i = 1:length(out_true)
    
        t_pred = squeeze(out_pred(i,:));
        t_true = squeeze(out_true(i,:));
        
        temp_acc = angularCorrCoeff(t_pred,t_true);
        acc_vec(i,1) = temp_acc;
        
    end
    
    disp(median(acc_vec))
    figure
    hist(acc_vec,100)
    xlabel('Angular Correlation Coefficient')
    ylabel('Number of Voxels')
    title('Shore 4 Shell Data Learning')


end