function visualize_vishabyte_multi_shell_shore_predictions_decayed_fod


    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\dl_results\Shore_input_Shore_decayed_output_v5\Vishabyte_Preds\result.mat')
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\shore_fod_decayed_basis_r6.mat')
    
    % Mask Path
    mask_path = 'D:\MASI LAB WORK\Solid_Harmonics\Solid_Harmonics\Vishabyte\128374\501\sk_stripped_mask.nii.gz';
    mask = load_untouch_nii(mask_path);
    mask = logical(mask.img);
    
    % Vishabyte Path
    nifti_path = 'D:\MASI LAB WORK\Solid_Harmonics\vishabyte_concat.nii.gz';
    nifti = load_untouch_nii(nifti_path);
    b0 = nifti.img(:,:,:,2);
    
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
    
    % Convert Shore to q-space for Vishabyte
    vish_preds = reshape(out_pred,[78 93 75 50]);
    vish_q_space = zeros(78,93,75,100);
    
    dims = size(vish_preds);
    
    for i = 1:dims(1)
        for j = 1:dims(2)
            for k = 1:dims(3)
                if (mask(i,j,k) == 1)
                    
                    shore_row = squeeze(vish_preds(i,j,k,:));
                    
                    if (size(shore_row,1) == 1)
                        shore_row = shore_row';
                    end
                    
                    q_space_pred_row = shore_basis * shore_row;
                    vish_q_space(i,j,k,:) = q_space_pred_row(2:101);
                    
                    
                end
            end
        end
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
    
    pred_sh = zeros(78,93,75,66);
    
    for i = 1:dims(1)
        for j = 1:dims(2)
            for k = 1:dims(3)
                if (mask(i,j,k) == 1)
                
                    pred_vox = squeeze(vish_q_space(i,j,k,:));
                    %pred_vox = pred_vox';
                    sh_pred_vox = (basis_sh10'*basis_sh10 + lambda*L_10)\basis_sh10'*squeeze(pred_vox);
                    pred_sh(i,j,k,:) = sh_pred_vox;
                    
                end
            end
        end
        if (mod(i,10)==0)
            disp(i)
        end
    end
    back_ground = pred_sh(:,:,:,1);
    back_ground(back_ground>2) = 0.5;
    
    % Visualize a middle axial slice of the predictions
    xform_RAS = eye(3);
    dv_single = dwmri_visualizer(pred_sh, ...  
    b0, ...
    mask, ...
    xform_RAS, ...    
    'sh_coefs', ...    
    {10,60,true});
    dv_single.plot_slice(39,'axial','slice');
    axis image;
    light('Position', [5, 5, 5], 'Style', 'infinite')
    title('Vishabyte FOD Predictions from 5 Shells')
    
    

end