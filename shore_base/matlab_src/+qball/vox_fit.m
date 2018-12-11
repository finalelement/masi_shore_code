function [sh_coeffs, exitcode] = vox_fit(dwi,bvecs,bvals,lmax,lambda)
    % Performs "Regularized, Fast and Robust Analytical Q-Ball" fit on a 
    % voxel.
    %
    % INPUTS:
    %   dwi are diffusion weighted values. Note that only single shell dwi
    %       should be input. No b0s or multishell dwi.
    %   Assumes size(bvecs) = [3 length(dwi)]. Also assumes bvecs are
    %       with respect to voxels.
    %   Assumes size(bvals) = [1 length(dwi)]
    %   lmax is the max spherical harmonic coefficient order
    %   lambda is the regularization coefficient
    %   
    % OUTPUTS:
    %   sh_coeffs are spherical harmonic coefficients 
    %   exitcode info in vol_fit
        
    % Just reshape and call the vol_fit since it is already vectorized.
    % Note that exitcode is set in vol_fit.
    [sh_coeffs, exitcode] = qball.vol_fit(reshape(dwi,1,1,1,length(dwi)),bvecs,bvals,lmax,lambda);
    sh_coeffs = reshape(sh_coeffs,1,[]);
end