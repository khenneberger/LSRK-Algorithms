%=========================================================
%
% DEMO DESTRIPING EXPERIMENTS FOR LSPK-S AND LSPK-K ALGORITHMS
%
% 
% MATLAB R2023b
% Author: Katherine Henneberger
% Institution: University of Kentucky - Math Department
%  
%=========================================================
clear
addpath(genpath(pwd));
rng(2023);

 
% load Face data
load('YaleFace.mat');
X = YaleFace./max(YaleFace(:));
[d1, d2 ,d3,d4] = size(X);
maxP = max(abs(X(:)));
maxP1 = max(YaleFace(:));
Nways=size(X);

% create A
A = zeros(d1,d1,d3,d4);
maxdiag = d1;
% create facewise diagonal tensor
for i = 1:maxdiag
    A(i,i,:,:) = 1;
end
% create stripes, as an example we create a stripe every 5th row
for i = 5:5:48
    A(i,i,:,:)=.01;
end
for i = 3:4
    A = ifft(A,[],i);
end

[n1, n2 ,n3,n4] = size(X);
Y = htprod_fft(A,X);

% initialize parameters
para.maxit = 500; 
para.bs = 1; 
para.numblock = 1;
para.control = 'cyc'; 
para.tol = 1e-2;
para.controltype = 'batch';
para.lambda = .1;
para.alpha = 1;
para.eps = 1;
para.gth = X;
para.type = 'lowrank';

%% Recovery
  fprintf('===== Destriping =====\n');
  t0=tic;
  out = LSPK_fft(A,Y,para);
  time = toc(t0); 
  Xhat1=max(0,out.X);
  Xhat2=min(maxP,Xhat1);
  err = out.err;
% assess the recovery

  Error = norm(Xhat2(:)-X(:))/norm(X(:));
  fprintf('Relative error = %0.8e\n',Error);
  psnr_index = PSNR(Xhat2,X,maxP);
  %[~,ssim,fsim]=quality(Xhat2*maxP1,X*maxP1);
  fprintf('PSNR = %0.8e\n',psnr_index);
    
% visualize the results

figure(1)
imagesc(X(:,:,11,10)); axis off; colormap gray; axis image;

figure(2);
imagesc(Y(:,:,11,10)); axis off; colormap gray; axis image;

figure(3);
imagesc(Xhat2(:,:,11,10)); axis off; colormap gray; axis image;



