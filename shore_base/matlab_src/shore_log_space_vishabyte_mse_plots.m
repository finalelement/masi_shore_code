function shore_log_space_vishabyte_mse_plots

    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\log_vishabyte_shore_reconst_r6.mat')
    
    % Mask Path
    mask_path = 'D:\MASI LAB WORK\Solid_Harmonics\Solid_Harmonics\Vishabyte\128374\501\sk_stripped_mask.nii.gz';
    mask = load_untouch_nii(mask_path);
    mask = logical(mask.img);

    % Load Original Data
    bval_path = 'D:\MASI LAB WORK\Solid_Harmonics\concat_bvals.bval';
    bvec_path = 'D:\MASI LAB WORK\Solid_Harmonics\concat_bvecs.bvec';
    nifti_path = 'D:\MASI LAB WORK\Solid_Harmonics\vishabyte_concat.nii.gz';
    nifti = load_untouch_nii(nifti_path);
    
    vish_reconst = exp(vish_reconst);
    mask_4d = repmat(mask,1,1,1,480);
    dims = size(mask);
    
    vish_reconst(~mask_4d) = 0;
    
    mse_b10 = zeros(78,93,75);
    mse_b15 = zeros(78,93,75);
    mse_b20 = zeros(78,93,75);
    mse_b25 = zeros(78,93,75);
    mse_b30 = zeros(78,93,75);
    
    % Separate out Predicted volumes into different b-vals and the same for
    % the original data
    orig_b10 = nifti.img(:,:,:,2:97);
    orig_b15 = nifti.img(:,:,:,98:193);
    orig_b20 = nifti.img(:,:,:,194:289);
    orig_b25 = nifti.img(:,:,:,290:385);
    orig_b30 = nifti.img(:,:,:,386:481);
    
    % Predicted Data
    pred_b10 = vish_reconst(:,:,:,2:97);
    pred_b15 = vish_reconst(:,:,:,98:193);
    pred_b20 = vish_reconst(:,:,:,194:289);
    pred_b25 = vish_reconst(:,:,:,290:385);
    pred_b30 = vish_reconst(:,:,:,386:481);

    % Calculate MSE
    for i = 1:dims(1)
        for j = 1:dims(2)
            for k = 1:dims(3)
                if (mask(i,j,k) == 1)
                    
                    mse_b10(i,j,k) = immse(double(orig_b10(i,j,k,:)),pred_b10(i,j,k,:));
                    mse_b15(i,j,k) = immse(double(orig_b15(i,j,k,:)),pred_b15(i,j,k,:));
                    mse_b20(i,j,k) = immse(double(orig_b20(i,j,k,:)),pred_b20(i,j,k,:));
                    mse_b25(i,j,k) = immse(double(orig_b25(i,j,k,:)),pred_b25(i,j,k,:));
                    mse_b30(i,j,k) = immse(double(orig_b30(i,j,k,:)),pred_b30(i,j,k,:));
                    
                end
            end
        end
    end
    
    % Lets plot some figures
    slice_num = 39;
    figure(1)
    subplot(1,5,1)
    imagesc(squeeze(mse_b10(:,:,slice_num)))
    axis tight 
    colorbar
    caxis([0 0.02])
    title('MSE b10')
    
    subplot(1,5,2)
    imagesc(squeeze(mse_b15(:,:,slice_num)))
    axis tight 
    colorbar
    caxis([0 0.02])
    title('MSE b15')
    
    subplot(1,5,3)
    imagesc(squeeze(mse_b20(:,:,slice_num)))
    axis tight 
    colorbar
    caxis([0 0.02])
    title('MSE b20')
    
    subplot(1,5,4)
    imagesc(squeeze(mse_b25(:,:,slice_num)))
    axis tight 
    colorbar
    caxis([0 0.02])
    title('MSE b25')
    
    subplot(1,5,5)
    imagesc(squeeze(mse_b30(:,:,slice_num)))
    axis tight 
    colorbar
    caxis([0 0.02])
    title('MSE b30')
     
end