function nsdn_sh_to_mag_to_ln_to_sh

    
    % Load Gradient Directions
    load('D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\py_code\NG_100.mat')
    
    % Load all the data
    load('D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\py_code\train_anchor_input.mat')
    load('D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\py_code\train_anchor_output.mat')
    load('D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\py_code\ta_voxels.mat')
    load('D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\py_code\tb_voxels.mat')
    
    % Set all the hyper-parameters
    lmax_dw = 8;
    lmax_fod = 10;
    lambda = 0.005;
    
    sphere_x = bvecs(1,:);
    sphere_y = bvecs(2,:);
    sphere_z = bvecs(3,:);
    [sh_sphere_dwmri,~,~] = construct_SH_basis(lmax_dw, [sphere_x(:) sphere_y(:) sphere_z(:)], 2, 'real');
    [sh_sphere_fod,~,~] = construct_SH_basis(lmax_fod, [sphere_x(:) sphere_y(:) sphere_z(:)], 2, 'real');

    sample_size = length(train_anchor_input);
    
    ip_sh_2_mag = [];
    op_sh_2_mag = [];
    ip_sh_2_ta_mag = [];
    ip_sh_2_tb_mag = [];
    
    disp('Transforming SH Coeffs to Magnitudes')
    
    for i=1:sample_size
    
        dw_row = train_anchor_input(i,:);
        fod_row = train_anchor_output(i,:);
        ta_row = ta_voxels(i,:);
        tb_row = tb_voxels(i,:);
        
        if (size(dw_row,1) == 1)
            dw_row = dw_row';
        end
        
        if (size(fod_row,1) == 1)
            fod_row = fod_row';
        end
        
        if (size(ta_row,1) == 1)
            ta_row = ta_row';
        end
        
        if (size(tb_row,1) == 1)
            tb_row = tb_row';
        end
        
        dw_b_row = sh_sphere_dwmri * dw_row;
        ta_b_row = sh_sphere_dwmri * ta_row;
        tb_b_row = sh_sphere_dwmri * tb_row;
        fod_b_row = sh_sphere_fod * fod_row;
        
        dw_b_row = dw_b_row';
        ta_b_row = ta_b_row';
        tb_b_row = tb_b_row';
        fod_b_row = fod_b_row';
        
        ip_sh_2_mag = [ip_sh_2_mag;dw_b_row];
        op_sh_2_mag = [op_sh_2_mag;fod_b_row];
        ip_sh_2_ta_mag = [ip_sh_2_ta_mag;ta_b_row];
        ip_sh_2_tb_mag = [ip_sh_2_tb_mag;tb_b_row];
        
        if(mod(i,1000)==0)
            disp(i)
        end
    end
    
    % Transform all magnitudes to log space
    ip_sh_2_mag(ip_sh_2_mag<=0) = lambda;
    op_sh_2_mag(op_sh_2_mag<=0) = lambda;
    ip_sh_2_ta_mag(ip_sh_2_ta_mag<=0) = lambda;
    ip_sh_2_tb_mag(ip_sh_2_tb_mag<=0) = lambda;
    
    ln_ip_sh_2_mag = log(ip_sh_2_mag)';
    ln_op_sh_2_mag = log(op_sh_2_mag)';
    ln_ip_sh_2_ta_mag = log(ip_sh_2_ta_mag)';
    ln_ip_sh_2_tb_mag = log(ip_sh_2_tb_mag)';
    
    % Fit Log magnitudes to SH space
    
    
    % Legendre Polynomial
    P0 = []; Laplac2 = [];
    for L_8=0:2:lmax_dw
        for m=-L_8:L_8
            Pnm = legendre(L_8, 0); factor1 = Pnm(1);
            P0 = [P0; factor1];
            Laplac2 = [Laplac2; (L_8^2)*(L_8 + 1)^2];
        end
    end
    L_8 = diag(Laplac2);
    
    % Legendre Polynomial
    P0 = []; Laplac2 = [];
    for L_10=0:2:lmax_fod
        for m=-L_10:L_10
            Pnm = legendre(L_10, 0); factor1 = Pnm(1);
            P0 = [P0; factor1];
            Laplac2 = [Laplac2; (L_10^2)*(L_10 + 1)^2];
        end
    end
    L_10 = diag(Laplac2);
    
    sh_ln_dwmri = [];
    sh_ln_ta = [];
    sh_ln_tb = [];
    sh_ln_fod = [];
    
    for i=1:sample_size
        
        ln_dw_row = ln_ip_sh_2_mag(:,i);
        ln_op_row = ln_op_sh_2_mag(:,i);
        ln_ta_row = ln_ip_sh_2_ta_mag(:,i);
        ln_tb_row = ln_ip_sh_2_tb_mag(:,i);
        
        sh_ln_dw_series = (sh_sphere_dwmri'*sh_sphere_dwmri + lambda*L_8)\sh_sphere_dwmri'*squeeze(ln_dw_row);
        sh_ln_ta_series = (sh_sphere_dwmri'*sh_sphere_dwmri + lambda*L_8)\sh_sphere_dwmri'*squeeze(ln_ta_row);
        sh_ln_tb_series = (sh_sphere_dwmri'*sh_sphere_dwmri + lambda*L_8)\sh_sphere_dwmri'*squeeze(ln_tb_row);
        sh_ln_fod_series = (sh_sphere_fod'*sh_sphere_fod + lambda*L_10)\sh_sphere_fod'*squeeze(ln_op_row);
        
        sh_ln_dw_series = sh_ln_dw_series';
        sh_ln_ta_series = sh_ln_ta_series';
        sh_ln_tb_series = sh_ln_tb_series';
        sh_ln_fod_series = sh_ln_fod_series';
        
        sh_ln_dwmri = [sh_ln_dwmri;sh_ln_dw_series];
        sh_ln_ta = [sh_ln_ta;sh_ln_ta_series];
        sh_ln_tb = [sh_ln_tb;sh_ln_tb_series];
        sh_ln_fod = [sh_ln_fod;sh_ln_fod_series];
        
        if (mod(i,1000)==0)
            disp(i)
        end
        
    end
    
    save('ln_train_anchor_input.mat','sh_ln_dwmri')
    save('ln_ta_voxels.mat','sh_ln_ta')
    save('ln_tb_voxels.mat','sh_ln_tb')
    save('ln_train_anchor_output.mat','sh_ln_fod')
    
end