function [b6k_acc_vec,b3k_b6k_acc_vec,b3k_b6k_b9k_acc_vec,b3k_b6k_b9k_b12k_acc_vec] = ipmi_2019_zeta_opt_eval_acc_log_v2
    
    base_result_path = 'D:\Users\Vishwesh\PycharmProjects\shore_mapmri\ipmi_results\zeta_optimized_log\';
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\shore_fod_decayed_basis_r6_log.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\out_matrix.mat')
    
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
    
    % Read the Shore Result files and concatenate them
    base_result_dir_list = dir(base_result_path);
    
    b6k_res = [];
    b3k_b6k_res = [];
    b3k_b6k_b9k_res = [];
    b3k_b6k_b9k_b12k_res = [];
    
    for i = 3:length(base_result_dir_list)
        
        b6k_path = fullfile(base_result_path,base_result_dir_list(i).name,'Hist_Blind_Test','b6k_test_result.mat');
        b3k_b6k_path = fullfile(base_result_path,base_result_dir_list(i).name,'Hist_Blind_Test','b3k_b6k_test_result.mat');
        b3k_b6k_b9k_path = fullfile(base_result_path,base_result_dir_list(i).name,'Hist_Blind_Test','b3k_b6k_b9k_test_result.mat');
        b3k_b6k_b9k_b12k_path = fullfile(base_result_path,base_result_dir_list(i).name,'Hist_Blind_Test','b3k_b6k_b9k_b12k_test_result.mat');
        
        load(b6k_path)
        b6k_res = [b6k_res;out_pred];
        clear out_pred
        
        load(b3k_b6k_path)
        b3k_b6k_res = [b3k_b6k_res;out_pred];
        clear out_pred
        
        load(b3k_b6k_b9k_path)
        b3k_b6k_b9k_res = [b3k_b6k_b9k_res;out_pred];
        clear out_pred
        
        load(b3k_b6k_b9k_b12k_path)
        b3k_b6k_b9k_b12k_res = [b3k_b6k_b9k_b12k_res;out_pred];
        clear out_pred
        
    end
    
    b6k_q_preds = zeros(length(out_matrix),100);
    b6k_q_full_preds = zeros(length(out_matrix),400);
    
    b3k_b6k_q_preds = zeros(length(out_matrix),100);
    b3k_b6k_q_full_preds = zeros(length(out_matrix),400);
    
    b3k_b6k_b9k_q_preds = zeros(length(out_matrix),100);
    b3k_b6k_b9k_q_full_preds = zeros(length(out_matrix),400);
    
    b3k_b6k_b9k_b12k_q_preds = zeros(length(out_matrix),100);
    b3k_b6k_b9k_b12k_q_full_preds = zeros(length(out_matrix),400);
    
    % Convert Shore to q-space for Pred
    for i = 1:length(out_matrix)
    
        % B6000
        b6k_shore_row = b6k_res(i,:);
        
        if (size(b6k_shore_row,1) == 1)
            b6k_shore_row = b6k_shore_row';
        end
        
        q_space_pred_row = shore_basis * b6k_shore_row;
        b6k_q_preds(i,:) = exp(q_space_pred_row(1:100));
        b6k_q_full_preds(i,:) = exp(q_space_pred_row);
        
        % B3000, B6000
        b3k_b6k_shore_row = b3k_b6k_res(i,:);
        
        if (size(b3k_b6k_shore_row,1) == 1)
            b3k_b6k_shore_row = b3k_b6k_shore_row';
        end
        
        q_space_pred_row = shore_basis * b3k_b6k_shore_row;
        b3k_b6k_q_preds(i,:) = exp(q_space_pred_row(1:100));
        b3k_b6k_q_full_preds(i,:) = exp(q_space_pred_row);
        
        % B3000, B6000, B9000
        b3k_b6k_b9k_shore_row = b3k_b6k_b9k_res(i,:);
        
        if (size(b3k_b6k_b9k_shore_row,1) == 1)
            b3k_b6k_b9k_shore_row = b3k_b6k_b9k_shore_row';
        end
        
        q_space_pred_row = shore_basis * b3k_b6k_b9k_shore_row;
        b3k_b6k_b9k_q_preds(i,:) = exp(q_space_pred_row(1:100));
        b3k_b6k_b9k_q_full_preds(i,:) = exp(q_space_pred_row);
        
        
        % B3000, B6000, B9000, B12000
        b3k_b6k_b9k_b12k_shore_row = b3k_b6k_b9k_b12k_res(i,:);
        
        if (size(b3k_b6k_b9k_b12k_shore_row,1) == 1)
            b3k_b6k_b9k_b12k_shore_row = b3k_b6k_b9k_b12k_shore_row';
        end
        
        q_space_pred_row = shore_basis * b3k_b6k_b9k_b12k_shore_row;
        b3k_b6k_b9k_b12k_q_preds(i,:) = exp(q_space_pred_row(1:100));
        b3k_b6k_b9k_b12k_q_full_preds(i,:) = exp(q_space_pred_row);
        
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
    b6k_pred_sh = zeros(length(out_matrix),66);
    b3k_b6k_pred_sh = zeros(length(out_matrix),66);
    b3k_b6k_b9k_pred_sh = zeros(length(out_matrix),66);
    b3k_b6k_b9k_b12k_pred_sh = zeros(length(out_matrix),66);
    
    for i = 1:length(out_matrix)
    
        pred_vox = b6k_q_preds(i,:);
        pred_vox = pred_vox';
        sh_pred_vox = (basis_sh10'*basis_sh10 + lambda*L_10)\basis_sh10'*squeeze(pred_vox);    
        b6k_pred_sh(i,:) = sh_pred_vox;
        
        pred_vox = b3k_b6k_q_preds(i,:);
        pred_vox = pred_vox';
        sh_pred_vox = (basis_sh10'*basis_sh10 + lambda*L_10)\basis_sh10'*squeeze(pred_vox);    
        b3k_b6k_pred_sh(i,:) = sh_pred_vox;
        
        pred_vox = b3k_b6k_b9k_q_preds(i,:);
        pred_vox = pred_vox';
        sh_pred_vox = (basis_sh10'*basis_sh10 + lambda*L_10)\basis_sh10'*squeeze(pred_vox);    
        b3k_b6k_b9k_pred_sh(i,:) = sh_pred_vox;
        
        pred_vox = b3k_b6k_b9k_b12k_q_preds(i,:);
        pred_vox = pred_vox';
        sh_pred_vox = (basis_sh10'*basis_sh10 + lambda*L_10)\basis_sh10'*squeeze(pred_vox);    
        b3k_b6k_b9k_b12k_pred_sh(i,:) = sh_pred_vox;
    
        if (mod(i,5000)==0)
            disp(i)
        end
    end
    
    b6k_acc_vec = zeros(length(out_matrix),1);
    b3k_b6k_acc_vec = zeros(length(out_matrix),1);
    b3k_b6k_b9k_acc_vec = zeros(length(out_matrix),1);
    b3k_b6k_b9k_b12k_acc_vec = zeros(length(out_matrix),1);
    
    for i = 1:length(out_matrix)
    
        t_true = squeeze(out_matrix(i,:));
        
        t_pred = squeeze(b6k_pred_sh(i,:));        
        temp_acc = angularCorrCoeff(t_pred,t_true);
        b6k_acc_vec(i,1) = temp_acc;
        
        t_pred = squeeze(b3k_b6k_pred_sh(i,:));        
        temp_acc = angularCorrCoeff(t_pred,t_true);
        b3k_b6k_acc_vec(i,1) = temp_acc;
        
        t_pred = squeeze(b3k_b6k_b9k_pred_sh(i,:));        
        temp_acc = angularCorrCoeff(t_pred,t_true);
        b3k_b6k_b9k_acc_vec(i,1) = temp_acc;
        
        t_pred = squeeze(b3k_b6k_b9k_b12k_pred_sh(i,:));        
        temp_acc = angularCorrCoeff(t_pred,t_true);
        b3k_b6k_b9k_b12k_acc_vec(i,1) = temp_acc;
        
    end
    
    b6k_med = median(b6k_acc_vec);
    b3k_b6k_med = median(b3k_b6k_acc_vec);
    b3k_b6k_b9k_med = median(b3k_b6k_b9k_acc_vec);
    b3k_b6k_b9k_b12k_med = median(b3k_b6k_b9k_b12k_acc_vec);
    
    title_1 = sprintf('Log B6000, Median: %0.4f',b6k_med);
    title_2 = sprintf('Log B3000,B6000, Median: %0.4f',b3k_b6k_med);
    title_3 = sprintf('Log B3000,B6000,B9000 Median: %0.4f',b3k_b6k_b9k_med);
    title_4 = sprintf('Log B3000,B6000,B9000,B12000 Median: %0.4f',b3k_b6k_b9k_b12k_med);
    
    bins = linspace(-1,1,200);
    
    figure
    subplot(2,2,1)
    hist(b6k_acc_vec,bins)
    xlabel('Angular Correlation Coefficient')
    ylabel('Number of Voxels')
    title(title_1)
    subplot(2,2,2)
    hist(b3k_b6k_acc_vec,bins)
    xlabel('Angular Correlation Coefficient')
    ylabel('Number of Voxels')
    title(title_2)
    subplot(2,2,3)
    hist(b3k_b6k_b9k_acc_vec,bins)
    xlabel('Angular Correlation Coefficient')
    ylabel('Number of Voxels')
    title(title_3)
    subplot(2,2,4)
    hist(b3k_b6k_b9k_b12k_acc_vec,bins)
    xlabel('Angular Correlation Coefficient')
    ylabel('Number of Voxels')
    title(title_4)
    
    
    
end