% Convert 4D DWMRI signal volume to SH volume 
function sh_vol = sh_vol_fit(nifti_vol,gradient,mask)
    
%     % Normalize by the b0
%     b0 = nifti_vol(:,:,:,1);
%     nifti_vol_grads = nifti_vol(:,:,:,2:97);
%     norm_vol = zeros(78,93,75,96);
%     
%     for i=1:96
%         temp = nifti_vol_grads(:,:,:,i)./b0;
%         norm_vol(:,:,:,i) = temp;
%     end

    lmax = 8;
    lambda = 0.005;  
    xform_RAS = eye(3);
    
    % Grab a single voxel
    %imp_signal = nifti_vol(35,65,37,:);
    bvec = gradient;
    
    %sh_sig = sh_basis(imp_signal,bvec);
    dims = size(nifti_vol);
    
    % SH-vol initialize
    sh_vol = zeros(dims(1),dims(2),dims(3),45);
    
    % Convert the entire volume to SH signal 
    for i=1:dims(1)
        for j=1:dims(2)
            for k=1:dims(3)
                if(mask(i,j,k)==1)
                    imp_signal = nifti_vol(i,j,k,:);
                    single_sh = sh_basis(imp_signal,bvec);                
                    % Store SH signal
                    sh_vol(i,j,k,:) = single_sh;
                end
            end
        end
        display(i)
    end
    
end
    %{
    % Visualize the SH signal
    re_sh = reshape(sh_sig,[1 1 1 45]);
    dv_single = dwmri_visualizer(re_sh, ...
                          1, ...
                          1, ...
                          xform_RAS, ...
                          'sh_coefs', ...
                          {8,120,true});
                      
    dv_single.plot_slice(1,'axial','slice');
    axis image;
    light('Position', [5, 5, 5], 'Style', 'infinite')
    title('Mid Slice voxel')
    %}                
    