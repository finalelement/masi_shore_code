function call_vol_sh_fit
    %nifti_path = 'D:\Users\Vishwesh\PycharmProjects\Deep_Linkage\vishabyte_b2000\test_9.nii';
    %nifti_path = 'D:\Users\Vishwesh\PycharmProjects\Deep_Linkage\vishabyte_b2000\b2000_avg_vishabyte.nii';
    %nifti_bvec = 'D:\Users\Vishwesh\PycharmProjects\Deep_Linkage\vishabyte_b2000\test_9.bvec';
    %global_mask = 'D:\Users\Vishwesh\PycharmProjects\Deep_Linkage\vishabyte_b2000\global_mask_mask.nii';
    %nifti_path = 'D:\Users\Vishwesh\PycharmProjects\Deep_Linkage\vishabyte_b2000\test_10.nii';
    %nifti_bvec = 'D:\Users\Vishwesh\PycharmProjects\Deep_Linkage\vishabyte_b2000\test_10.bvec';
    
    % TS 00 Paths
    %global_mask = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\T1_WM\ts_wm_mask.nii';
    %nifti_path = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\3TB\b2000_ts00_3tb.nii';
    %nifti_bvec = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\3TB\b2000_bvecs_3tb.bvec';
    
    %TS 04 Paths
    %global_mask = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\TS04\T1_WM\ts04_wm_mask.nii';
    %nifti_path = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\TS04\3TB\b_ng_dwmri.nii';
    %nifti_bvec = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\TS04\3TB\b_ng_bvec.bvec';
    
    %TS 01 Paths
    global_mask = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\TS01\T1_WM\ts01_wm_mask.nii';
    nifti_path = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\TS01\Austin\austin_ng_dwmri.nii';
    nifti_bvec = 'D:\Users\Vishwesh\PycharmProjects\Deep_Null_Space\Final_data\TS01\Austin\austin_ng_bvec.bvec';
    
    nifti = load_untouch_nii(nifti_path);
    nifti_bvec = dlmread(nifti_bvec);
    %nifti_bvec = nifti_bvec(:,2:97);
    mask = load_untouch_nii(global_mask);
    mask = mask.img;
    
    %{ 
    Removing it specifically for TS04 because it is normalized data.
    
    % Divide the entire gradient volume by b0 before passing it in
    my_img = nifti.img(:,:,:,2:97);
    b0 = nifti.img(:,:,:,1);
    dims = size(my_img);
    
    norm_nifti = zeros(dims(1),dims(2),dims(3),96);
    
    for i=1:96
        temp = my_img(:,:,:,i)./b0;
        norm_nifti(:,:,:,i) = temp;
    end
    %}
    
    
    sh_vol = sh_vol_fit(nifti.img,nifti_bvec,mask);
    sh_vol(isnan(sh_vol)) = 0;
    
end