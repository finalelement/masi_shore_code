
%%=============================================================
%% Project:   Spherical Harmonics
%% Module:    $RCSfile: test_rotation.m,v $
%% Language:  MATLAB
%% Author:    $Author: bjian $
%% Date:      $Date: 2007/12/27 06:23:36 $
%% Version:   $Revision: 1.5 $
%%=============================================================

A = rand(3);
[U,S,V] = svd(A);
if det(U)<0
    r = -U;
else
    r = U;
end


degree = 6;
v = load('321vectors.txt');
Y = qball.lib.spherical_harmonics.construct_SH_basis(degree,v);

coeff = rand(10000,49);


dest = qball.lib.spherical_harmonics.rotate_coeff(coeff, r, degree);

vv = v*r';
YY = qball.lib.spherical_harmonics.construct_SH_basis(degree,vv);

norm(Y*coeff' - YY*dest','inf')



