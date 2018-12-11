function evaluate_shore_dl_preds_shore_non_negative_decayed_output

    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Shore_4_shells_trial_1\testing_results\test_results.mat')
    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Shore_4_shells_trial_3_r8\Hist_Blind_72_Test\result.mat')
    
    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Mapmri_4_shells_trial_1_r6\Hist_Blind_72_Test\result.mat')
    
    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Mapmri_4_shells_trial_2_r8\Hist_Blind_72_Test\result.mat')
    %load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Shore_input_Shore_output_v1\Hist_Blind_72_Test\result.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Non_Neg_Shore_input_Shore_decayed_output_v4\Hist_Blind_72_Test\result.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\shore_fod_decayed_basis_r6_log.mat')
    
    % Load Gradient Directions
    load('D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\py_code\NG_100.mat')
    bvec = bvecs;
    if any(size(bvec)==3)
        if size(bvec,2) ~=3
            req_bvecs = bvec';
        else
            req_bvecs = bvec;
        end
    else
        error('DIRECTIONS ARE NOT DEFINED ON R3')
    end
    
    
    acc_vec = zeros(length(out_true),1);

    % Convert Shore to q-space for both true and pred
    q_preds = zeros(length(out_true),100);
    q_true = zeros(length(out_true),100);
    
    q_full_preds = zeros(length(out_true),400);
    q_full_true = zeros(length(out_true),400);
    
    for i = 1:length(out_true)
    
        shore_row = out_pred(i,:);
        shore_t_row = out_true(i,:);
        
        if (size(shore_row,1) == 1)
            shore_row = shore_row';
        end
        
        if (size(shore_t_row,1) == 1)
            shore_t_row = shore_t_row';
        end
        
        q_space_pred_row = shore_basis * shore_row;
        q_preds(i,:) = exp(q_space_pred_row(1:100));
        q_full_preds(i,:) = exp(q_space_pred_row);
        
        q_space_true_row = shore_basis * shore_t_row;
        q_true(i,:) = exp(q_space_true_row(1:100));
        q_full_true(i,:) = exp(q_space_true_row);
        
    end
    
    % Fit q-space True and Predictions to Spherical Harmonics
    lmax = 10;
    lambda = 0.005;
    
    % Legendre Polynomial
    P0 = []; Laplac2 = [];
    for L_10=0:2:lmax
        for m=-L_10:L_10
            Pnm = legendre(L_10, 0); factor1 = Pnm(1);
            P0 = [P0; factor1];
            Laplac2 = [Laplac2; (L_10^2)*(L_10 + 1)^2];
        end
    end
    L_10 = diag(Laplac2);
    [basis_sh10,~,~] = construct_SH_basis(lmax,req_bvecs,2,'real');
    
    % Initialize SH variables
    pred_sh = zeros(length(out_true),66);
    true_sh = zeros(length(out_true),66);
    
    for i = 1:length(out_true)
    
        pred_vox = q_preds(i,:);
        true_vox = q_true(i,:);
        
        pred_vox = pred_vox';
        true_vox = true_vox';
        
        sh_pred_vox = (basis_sh10'*basis_sh10 + lambda*L_10)\basis_sh10'*squeeze(pred_vox);
        sh_true_vox = (basis_sh10'*basis_sh10 + lambda*L_10)\basis_sh10'*squeeze(true_vox);
        
        pred_sh(i,:) = sh_pred_vox;
        true_sh(i,:) = sh_true_vox;
        
    end
    
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\out_matrix.mat')
    true_sh = out_matrix(49996:end,:);
    
    for i = 1:length(out_true)
    
        t_pred = squeeze(pred_sh(i,:));
        t_true = squeeze(true_sh(i,:));
        
        temp_acc = angularCorrCoeff(t_pred,t_true);
        acc_vec(i,1) = temp_acc;
        
    end
    
    disp(median(acc_vec))
    figure
    hist(acc_vec,100)
    xlabel('Angular Correlation Coefficient')
    ylabel('Number of Voxels')
    grid on
    title('Shore 4 Shell - Log Space Decayed FOD')
    
    xform_RAS = eye(3);
    vox_num = 3512;
    true_vox = true_sh(vox_num,:);
    pred_vox = pred_sh(vox_num,:);
    
    re_tr_vox = reshape(true_vox,[1 1 1 66]);
    re_pr_vox = reshape(pred_vox,[1 1 1 66]);
    
    dv_single = dwmri_visualizer(re_tr_vox, ...  
    ones(1,1,1), ...
    logical(ones(1,1,1)), ...
    xform_RAS, ...    
    'sh_coefs', ...    
    {10,60,true});
    dv_single.plot_slice(1,'axial','slice');
    axis image;
    light('Position', [5, 5, 5], 'Style', 'infinite')
    title('True Voxel')
    
    dv_single = dwmri_visualizer(re_pr_vox, ...  
    ones(1,1,1), ...
    logical(ones(1,1,1)), ...
    xform_RAS, ...    
    'sh_coefs', ...    
    {10,60,true});
    dv_single.plot_slice(1,'axial','slice');
    axis image;
    light('Position', [5, 5, 5], 'Style', 'infinite')
    title('Pred Voxel')
    
end