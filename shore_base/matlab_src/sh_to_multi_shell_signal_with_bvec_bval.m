function sh_to_multi_shell_signal_with_bvec_bval

    % Load Gradient Directions
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\NG_100.mat')
    
    % Load the input data
    load('D:\Users\Vishwesh\PycharmProjects\shore_mapmri\Data\input_matrix.mat')
    
    % Multi-shell matrix initialization
    data_dims = size(input_matrix);
    ms_data = zeros(data_dims(1), 400);
    ms_bvecs = zeros(3,400);
    ms_bvals = zeros(1,400);
    
    % Assign bval values
    ms_bvals(:,1:100) = 3000;
    ms_bvals(:,101:200) = 6000;
    ms_bvals(:,201:300) = 9000;
    ms_bvals(:,301:400) = 12000;
    
    % Assign bvec values
    ms_bvecs(:,1:100) = bvecs;
    ms_bvecs(:,101:200) = bvecs;
    ms_bvecs(:,201:300) = bvecs;
    ms_bvecs(:,301:400) = bvecs;
    
    % Construct the basis function for reconstructing the signal
    lmax_dw = 8;
    lambda = 0.005;
    
    sphere_x = bvecs(1,:);
    sphere_y = bvecs(2,:);
    sphere_z = bvecs(3,:);

    [sh_sphere_dwmri,~,~] = construct_SH_basis(lmax_dw, [sphere_x(:) sphere_y(:) sphere_z(:)], 2, 'real');
    
    disp('Transforming SH Coeffs to Magnitudes')
    
    sample_size = data_dims(1);
    
    for i=1:sample_size
    
        b3k_row = input_matrix(i,1:45);
        b6k_row = input_matrix(i,46:90);
        b9k_row = input_matrix(i,91:135);
        b12k_row = input_matrix(i,136:180);
        
        if (size(b3k_row,1) == 1)
            b3k_row = b3k_row';
        end
        
        if (size(b6k_row,1) == 1)
            b6k_row = b6k_row';
        end
        
        if (size(b9k_row,1) == 1)
            b9k_row = b9k_row';
        end
        
        if (size(b12k_row,1) == 1)
            b12k_row = b12k_row';
        end
        
        b3k_mag_row = sh_sphere_dwmri * b3k_row;
        b6k_mag_row = sh_sphere_dwmri * b6k_row;
        b9k_mag_row = sh_sphere_dwmri * b9k_row;
        b12k_mag_row = sh_sphere_dwmri * b12k_row;
        
        b3k_mag_row = b3k_mag_row';
        b6k_mag_row = b6k_mag_row';
        b9k_mag_row = b9k_mag_row';
        b12k_mag_row = b12k_mag_row';
        
        ms_data(i,1:100) = b3k_mag_row;
        ms_data(i,101:200) = b6k_mag_row;
        ms_data(i,201:300) = b9k_mag_row;
        ms_data(i,301:400) = b12k_mag_row;
          
    end

    dlmwrite('ms_bvals.bval',ms_bvals)
    dlmwrite('ms_bvecs.bvec',ms_bvecs)
    save('multi_shell_data.mat','ms_data')
    
end