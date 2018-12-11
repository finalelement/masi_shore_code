function [sh_coeffs_vol, exitcode_vol] = vol_fit(dwi_vol,bvecs,bvals,lmax,lambda,mask_vol)
    % Performs "Regularized, Fast and Robust Analytical Q-Ball" fit on a 
    % dwi volume.
    %
    % INPUTS:
    %   dwi_vol are the diffusion weighted volumes. Assumes diffusion
    %       weightings are done along the 4th dimension
    %   Assumes size(bvecs) = [3 size(dwi_vol,4)]. Also assumes bvecs are
    %       with respect to voxels.
    %   Assumes size(bvals) = [1 size(dwi_vol,4)]
    %   lmax is the max spherical harmonic coefficient order
    %   lambda is the regularization coefficient - 0.006
    %   mask_vol is optional
    %   
    % OUTPUTS:
    %   sh_coeffs_vol are spherical harmonic coefficients 
    %   exitcode_vol:
    %   exitcode of -1 = background data
    %   exitcode of 0 = fit was successful
    %   exitcode of 1 = non-finite value in spherical harmonic coefficients
    
    if any(bvals == 0)
        error('Only diffusion weighted images. No b0s for q-ball.');
    end
    
    if length(unique(bvals)) ~= 1
        error('Only single-shell DWI is supported by "Regularized, Fast and Robust Analytical Q-Ball"');
    end
    
    if ~exist('mask_vol','var') || isempty(mask_vol)
        mask_vol = true(size(dwi_vol,1),size(dwi_vol,2),size(dwi_vol,3));
    end
    
    % Get spherical harmonic coefficients
    sh_coeffs_vol = qball.lib.QBALL_RECON_v1(dwi_vol,bvecs',lmax,lambda);
    
    % Clear out values outside mask (don't worry about using the mask 
    % during computation; this fit is fast without it since it is 
    % completely vectorized).
    sh_coeffs_vol(~repmat(mask_vol,1,1,1,size(sh_coeffs_vol,4))) = 0;

    % Set exitcode
    exitcode_vol = -ones(size(mask_vol));    
    exitcode_vol(mask_vol) = 0;
    exitcode_vol(any(~isfinite(sh_coeffs_vol),4)) = 1;
end