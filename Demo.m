%**************************************************************************
% Author: Oumaima El Mansouri (2019 Oct.)
% University of Toulouse, IRIT/INP-ENSEEIHT
% Email: oumaima.el-mansouri@irit.fr
% ---------------------------------------------------------------------
% Copyright (2020): Oumaima El Mansouri, Adrian Basarab, Denis Kouamé, Jean-Yves Tourneret.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------
% 
% This set of MATLAB files contain an implementation of the algorithms
% described in the following paper (Without backtracking step):
% 
% [1] O. E. Mansouri, A. Basarab, F. Vidal, D. Kouamé and J. Tourneret, "Fusion Of Magnetic Resonance And Ultrasound 
%  Images: A Preliminary Study On Simulated Data," 2019 IEEE 16th International Symposium on Biomedical Imaging 
% (ISBI 2019),Venice, Italy, 2019, pp. 1733-1736, doi: 10.1109/ISBI.2019.8759524.
%
% ---------------------------------------------------------------------
%************************************************************************** 
%%
close all
clear all
clc

addpath ./utils;
addpath ./images;

%% Load or read images
% if needed resize MRI and US images (Nus = d*Nmri), in this example d = 2
% (d is an integer)
load('y1')
load('y2')

%% Image normalization
%linear normalization
ym = double(y1)./double(max(y1(:)));
yu = double(y2)./double(max(y2(:)));

%% Display observations

figure; imshow(ym,[]);
figure; imshow(yu,[]);

%% Initialization of ADMM

d=2; %MRI and US must have the same size
x1 = imresize(ym,d,'bicubic'); %MRI bicubic interpolation
x2 = yu + 1e-8; 

%compute MRI gradient
Jx = conv2(x1,[-1 1],'same');
Jy = conv2(x1,[-1 1]','same');
gradY = sqrt(Jx.^2+Jy.^2);
%% Regularization parameters
sigma = 0.0017;
taup = 5e6;
tau = taup*sigma^2;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% paramètre IRM (influence TV) %%%%%%%%%%%%%%%%%%%%%%%%%
tau10 = 1;        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% paramètre IRM (influence echo) %%%%%%%%%%%%%%%%%%%%%%%%%
tau1 = 1e-5;       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% paramètre US (influence observation) %%%%%%%%%%%%%%%%%%%%%%%%%
tau2 = 1e-3;       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% paramètre US (influence TV) %%%%%%%%%%%%%%%%%%%%%%%%%
tau3 = 2e-2;       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% paramètre US (influence IRM) %%%%%%%%%%%%%%%%%%%%%%%%%
c2 = 0.05;
%%
[x2] = linear_ADMM(y1,y2,x1,x2,gradY,c2,d,tau,tau10,tau1, tau2, tau3, true, true);

function [x2] = linear_ADMM(y1,y2,x1,x2,gradY,c2,d,tau,tau10,tau1, tau2, tau3, plot_fused_image, plot_criterion)
%% parameters

% define the difference operator kernel
[n1,n2] = size(x1);
dh = zeros(n1,n2);
dh(1,1) = 1;
dh(1,2) = -1;
dv = zeros(n1,n2);
dv(1,1) = 1;
dv(2,1) = -1;
% compute FFTs for filtering
FDH = fft2(dh);
F2DH = abs(FDH).^2;
FDV = fft2(dv);
FDV = conj(FDV);
F2DV = abs(FDV).^2;
c1 = 1e-8;
F2D = F2DH + F2DV +c1;
alpha = c1 +c2*gradY;
gama = 1e-3;
%% ADMM
maxiter = 10;
D = zeros(n1,n2);
B = fspecial('gaussian',7,3);
[FB,FBC,F2B,Bx] = HXconv(x1,B,'Hx');
BX = ifft2(FB.*fft2(x1));
resid1 =  y1 - BX(1:d:end,1:d:end);
resid2 = ifft2(F2D.*fft2(x1));
resid3 =sum(sum(gama*exp(y2-x2)-(y2-x2)));
resid4 = (x2-c2*x1 - alpha);
objective(1) = 0.5*(resid1(:)'*resid1(:)) +  tau*(resid2(:)'*resid2(:)) + tau1*resid3 + tau2*(norm(d1(x2))^2)+tau3*(resid4(:)'*resid4(:));
for i = 1:maxiter
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% update Xirm %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    x1 = FSR_xirm(y1,x2,D,alpha,B,d,c2,F2D,tau,tau10,'false');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% update Xus %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    x2 = Descente_grad_xus(y2,x1,D,alpha,c2,gama,tau1,tau2,tau3,'false');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% update D %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    D = D + tau3*(x2-c2*x1 - alpha);
    BX = ifft2(FB.*fft2(x1));
    resid1 =  y1 - BX(1:d:end,1:d:end);
    resid2 = ifft2(F2D.*fft2(x1));
    resid3 =sum(sum(gama*exp(y2-x2)-(y2-x2)));
    resid4 = (x2-c2*x1 - alpha);
    objective(i+1) = 0.5*(resid1(:)'*resid1(:)) +  tau*(resid2(:)'*resid2(:)) + tau1*resid3 + tau2*(norm(d1(x2))^2)+tau3*(resid4(:)'*resid4(:));
    criterion(i) = abs(objective(i+1)-objective(i))/objective(i);
end
%% Display

if plot_fused_image
    figure; imshow(x1,[]);
end
if plot_criterion
    figure; plot(criterion);
    title('Courbe de convergence ADMM (contrainte non-linéaire)')
    xlabel('nombre d itération')
    ylabel('résidu')
end
end