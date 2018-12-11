function sh_signal = sh_basis(req_4d_dwmri,bvec)
   
 
    req_bvecs = bvec;

    lmax = 8;
    lambda = 0.005;
    
    % Legendre Polynomial
    P0 = []; Laplac2 = [];
    for L=0:2:lmax
        for m=-L:L
            Pnm = legendre(L, 0); factor1 = Pnm(1);
            P0 = [P0; factor1];
            Laplac2 = [Laplac2; (L^2)*(L + 1)^2];
        end
    end
    L = diag(Laplac2);
    [basis,~,~] = qball.lib.spherical_harmonics.construct_SH_basis(lmax,req_bvecs',2,'real');
    dwmri_op = req_4d_dwmri;
    
    % Reconstruct signal with spherical harmonic coefficients using regularized fit                   
    sh_signal = (basis'*basis + lambda*L)\basis'*squeeze(dwmri_op);                  
    % Reconstruct signal
    %sh_signal = basis*C;
    
    
    %w = ones(64,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BL - Experiment
    %{
    residual = squeeze(dwmri_op) - squeeze(sh_signal);
    sigma = 0.0453;
    w = 1./(residual.^2 + sigma^2);
    %w = ones(64,1) * 10;
    
    wtdwmri_op = w.*squeeze(dwmri_op);
    wtbasis = repmat(w,[1 size(basis,2)]).*basis;
    %pinvwtbasis = pinv(wtbasis);
    robustC = (wtbasis'*wtbasis + lambda*L)\(wtbasis'*squeeze(wtdwmri_op));                  
    %robustC = pinv(wtbasis)*squeeze(wtdwmri_op);
    wsh_signal = basis*robustC;
   
    figure
    plot(sh_signal)
    hold on
    plot(wsh_signal)
    %}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
end