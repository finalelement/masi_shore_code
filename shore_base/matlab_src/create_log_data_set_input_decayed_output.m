function create_log_data_set_input_decayed_output

    
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\multi_shell_data_b0.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.mat')
    
    ln_ms_data = log(ms_data(:,2:end));
    ln_fod_data = log(new_fod_q_space(:,2:end));

    ln_dwmri_data = zeros(length(ms_data),400);
    ln_decayed_fod_data = zeros(length(new_fod_q_space),400);
    
    %ln_dwmri_data(:,1) = 1;
    %ln_decayed_fod_data(:,1) = 1;
    
    ln_dwmri_data(:,:) = ln_ms_data;
    ln_decayed_fod_data(:,:) = ln_fod_data;

    save('ln_ms_data.mat','ln_dwmri_data')
    save('ln_decayed_fod.mat','ln_decayed_fod_data')

    % Load dwmri bval and bvecs
    dw_bval = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals_b0.bval');
    dw_bvec = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs_b0.bvec');
    
    fod_bval = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bval');
    fod_bvec = dlmread('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod.bvec');
    
    dw_bval(1) = [];
    dw_bvec(:,1) = [];
    
    fod_bval(1) = [];
    fod_bvec(:,1) = [];
    
    % Save the new bvals and the bvecs
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvals.bval',dw_bval);
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ms_bvecs.bvec',dw_bvec);
    
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod_bval.bval',fod_bval);
    dlmwrite('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\decayed_fod_bvec.bvec',fod_bvec);
    
end