function  plm_4d = QBALL_RECON_v1(dwi4d, grad, Lmax, Lambda)
% QBALL_RECON - performs QBI analysis on DWI data
% See Descoteaux "Regularized, Fast, and Robust Analytical Q-Ball Imaging"
% Code adapted from NeuroimageN (HARDI) Toolbox
%
%
% Inputs:
%    dwi4d: 4d DWI's (rows*cols*slices*N matrix)
%    grad: gradient table (N*3 matrix)
%    Lmax: The maximum order of the spherical harmonics
%    lambda: regularization amount
%
% Outputs: plm_4d: spherical Harmonic coefficients for all voxels (stored in standard order)
%
% Other m-files required : none
% Subfunctions (from NeuroimageN): construct_SH_basis, legendre, recon_matrix
%
% Author:  Schilling, K
% Date:    15-Jan-2016
% Version: 1.0
% Changelog:
%
% 15-Jan-2016 - initial creation

    %%%%%%%%%%%%%%%%%%%% BEGIN CODE %%%%%%%%%%%%%%%%%%%%
    h = sprintf('DWI volume: Rows=%.0f Cols=%.0f Slices=%.0f Directions=%.0f', size(dwi4d, 1), size(dwi4d, 2), size(dwi4d, 3), size(dwi4d, 4)); 
    disp(h);

    % Remove the S0 component from the signal (if exists), which is not necessary for Q-Ball Imaging
    grad_sum = sum(abs(grad), 2);
    ind_0 = find(grad_sum == 0);
    dwi4d(:, :, :, ind_0) = [];
    grad(ind_0, :) = [];

    % real spherical harmonic reconstruction
    display(['The maximum order of the spherical harmonics is Lmax = ' num2str(Lmax)]);
    display(['The regularization parameter Lambda = ' num2str(Lambda)]);

    [basisG, thetaG, phiG] = qball.lib.spherical_harmonics.construct_SH_basis(Lmax, grad, 2, 'real');

    P0 = []; Laplac2 = [];
    for L=0:2:Lmax
        for m=-L:L
            Pnm = legendre(L, 0); factor1 = Pnm(1);
            P0 = [P0; factor1];
            Laplac2 = [Laplac2; (L^2)*(L + 1)^2];
        end
    end
    L = diag(Laplac2);

    %  Creating the kernel for the reconstruction
    A = (basisG'*basisG + Lambda*L)\basisG';
    plm_4d = zeros(size(dwi4d, 1), size(dwi4d, 2), size(dwi4d, 3), length(P0));

    for i=1:length(P0)
        for j=1:size(dwi4d, 4)
            plm_4d(:, :, :, i) =  plm_4d(:, :, :, i) + A(i, j)*squeeze(dwi4d(:, :, :, j))*P0(i);
        end
    end

end



