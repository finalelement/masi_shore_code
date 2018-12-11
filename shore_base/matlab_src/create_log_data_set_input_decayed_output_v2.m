function create_log_data_set_input_decayed_output_v2

    log_lambda = 0.999;

    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\multi_shell_data_b0.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.mat')
    
    ln_ms_data = log(ms_data(:,1:400));
    ln_fod_data = log(new_fod_q_space(:,2:end));

    ln_dwmri_data = zeros(length(ms_data),401);
    ln_decayed_fod_data = zeros(length(new_fod_q_space),401);
    
    %ln_dwmri_data(:,1) = 1;
    %ln_decayed_fod_data(:,1) = 1;
    
    ln_dwmri_data(:,1:400) = ln_ms_data;
    ln_decayed_fod_data(:,1:400) = ln_fod_data;

    ln_dwmri_data(:,401) = log(log_lambda);
    ln_decayed_fod_data(:,401) = log(log_lambda);
    
    save('ln_ms_data.mat','ln_dwmri_data')
    save('ln_decayed_fod.mat','ln_decayed_fod_data')

    % Load dwmri bval and bvecs
    dw_bval = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals_b0.bval');
    dw_bvec = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs_b0.bvec');
    
    fod_bval = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bval');
    fod_bvec = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bvec');
    
    %dw_bval(1) = [];
    %dw_bvec(:,1) = [];
    
    t_fod_bval = fod_bval(2:401);
    t_fod_bval(401) = 0;
    t_fod_bvec = fod_bvec(:,2:401);
    t_fod_bvec(:,401) = 0;
    
    % Save the new bvals and the bvecs
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals.bval',dw_bval);
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs.bvec',dw_bvec);
    
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod_bval.bval',t_fod_bval);
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod_bvec.bvec',t_fod_bvec);
    
end