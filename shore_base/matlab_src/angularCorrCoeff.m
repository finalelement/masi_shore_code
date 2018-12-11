function acc = angularCorrCoeff(p_v, q_v)
% Usage: acc = angularCorrCoeff(p_v, q_v)
%
% angularCorrCoeff calculates the correlation coefficient
% of two functions, P(theta, phi) and Q(theta, phi), over
% all orientations, based on their spherical harmonic 
% decompositions. The coefficients of the spherical harmonic 
% expansions of P and Q are p_v and q_v, respectively 
% (coefficients are given in 'standard order', i.e., 
% p_v(1) is the (0,0) coefficient, p_v(2) is the (1,-1)
% coefficient, and so on). 
%

% Preliminaries:

% Make sure vectors have the same length:
np = length(p_v);
nq = length(q_v);
nCoeffs = min([np, nq]);
p_v = p_v(1:nCoeffs);
q_v = q_v(1:nCoeffs);

% Subtract the mean value from the expansions of P and Q:
zmp_v = p_v(2:nCoeffs);
zmq_v = q_v(2:nCoeffs);

% Normalize each vector:
np_v = zmp_v / sqrt(sum(zmp_v .* conj(zmp_v)));
nq_v = zmq_v / sqrt(sum(zmq_v .* conj(zmq_v)));

% Calculate correlation coefficient:
acc = real(sum(np_v .* conj(nq_v)));