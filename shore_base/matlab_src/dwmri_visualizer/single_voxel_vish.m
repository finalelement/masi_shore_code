%% Set environment
addpath(genpath('/home/local/VANDERBILT/blaberj/masimatlab/trunk/users/blaberj/matlab/justinlib_v1_7_0'));
addpath('/home/local/VANDERBILT/blaberj/masimatlab/trunk/users/blaberj/dwmri_libraries/dwmri_visualizer_v1_2_0/');
addpath('/home/local/VANDERBILT/blaberj/NIFTI');
addpath('/home/local/VANDERBILT/blaberj/spherical_harmonics');

%% Single voxel
clear

% Load data
dtiqa_path = '/fs4/masi/blaberj/data/test_spider/LANDMAN_UPGRAD-x-7LC4HTKA-x-7LC4HTKA-x-dtiQA_v5/';
qball_path = '/fs4/masi/blaberj/data/test_spider/LANDMAN_UPGRAD-x-7LC4HTKA-x-7LC4HTKA-x-Qball_v1';
sh_coefs_vol = nifti_utils.load_untouch_nii4D_vol_scaled(fullfile(qball_path,'/SH_COEFS/qball_sh_coefs.nii.gz'),'double');
fa_vol = nifti_utils.load_untouch_nii_vol_scaled(fullfile(dtiqa_path,'/RESTORE/fa.nii.gz'),'double');
mask_vol = nifti_utils.load_untouch_nii_vol_scaled(fullfile(dtiqa_path,'/MASK/b0_all_mask_mask.nii.gz'),'logical');
xform_RAS = nifti_utils.get_voxel_RAS_xform(fullfile(dtiqa_path,'/PREPROCESSED/dwmri.nii.gz'));

% Get single voxel
sh_coefs = sh_coefs_vol(45,45,25,:);

% Get visualizer
dv = dwmri_visualizer(sh_coefs, ...
                      1, ...
                      1, ...
                      xform_RAS, ...
                      'sh_coefs', ...
                      {6,120,true});

% Make a plot
dv.plot_slice(1,'axial','slice');
axis image;
light('Position', [5, 5, 5], 'Style', 'infinite')