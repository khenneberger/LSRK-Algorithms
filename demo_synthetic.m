%=========================================================
%
% DEMO SYNTHETIC EXPERIMENTS FOR LSPK-S AND LSPK-K ALGORITHMS
%
% 
% MATLAB R2023b
% Author: Katherine Henneberger
% Institution: University of Kentucky - Math Department
%  
%=========================================================


addpath(genpath(pwd))

% Generate random tensors
n = 10;
rng(2023);
A=randn(n,2,n,n)/n;
rng(2023);
X=randn(2,n,n,n)/n;
Nways=size(X);

% create Y as the high-order t-product using fft
Y = htprod_fft(A,X);

% set up parameters
maxx        = 1000; % max number of iterations
para.maxit  = maxx;
para.bs     = 7; % tradeoff between accuracy and speed
para.gth    = X; % ground truth
para.alpha  = 1; % stepsize
para.eps    = .1;
para.lambda = .001;
para.numblock = 3;
para.type   = "lowrank"; % "sparse" runs LSPK-S and "lowrank" runs LSPK-L

%% set number of trials and run algorithm
trials = 1;

storeblockrand    = zeros(trials,maxx);
storeblockcyc     = zeros(trials,maxx);
storebatchrand    = zeros(trials,maxx);
storebatchcyc     = zeros(trials,maxx);
for i = 1:trials
    disp(i); %track trials

    % block rand
    para.control = 'rand'; 
    para.controltype = 'block';
    outblockrand = LSPK_fft(A,Y,para);
    storeblockrand(i,:) = outblockrand.err(1:maxx);
 
    % block cyc
    para.control = 'cyc';
    para.controltype = 'block';
    outblockcyc = LSPK_fft(A,Y,para);
    storeblockcyc(i,:) = outblockcyc.err(1:maxx);
 
    % batch rand
    para.control = 'rand'; 
    para.controltype = 'batch';
    outbatchrand = LSPK_fft(A,Y,para);
    storebatchrand(i,:) = outbatchrand.err(1:maxx);
 

    % batch cyc
    para.control = 'cyc'; 
    para.controltype = 'batch';
    outbatchcyc = LSPK_fft(A,Y,para);
    storebatchcyc(i,:) = outbatchcyc.err(1:maxx);
 
end

% average over all trials & save results
trialavgblr  = mean(storeblockrand,1);
trialavgblc  = mean(storeblockcyc,1);
trialavgbtr  = mean(storebatchrand,1);
trialavgbtc  = mean(storebatchcyc,1);

%% plot

figure()
line_types = { '--o','-v', '-*','-.*','-.x'};
marker_color = ["#4DBEEE","#000000","#77AC30","red","blue"];
line_types_cnt = size(line_types, 2);
line_width = 1.5;
marker_size = 6;
x = 1:50:maxx;  % plot every 50th result
selecty = zeros(size(x,2),4);
cnt = 1;
for j = 1:50:maxx
    selecty(cnt,1) = trialavgblr(j);
    selecty(cnt,2) = trialavgblc(j);
    selecty(cnt,3) = trialavgbtr(j);
    selecty(cnt,4) = trialavgbtc(j);
    cnt = cnt+1;
end
for i =  1:4
    semilogy(x,selecty(:,i),line_types{1, mod(i, line_types_cnt)+1}, 'Color', ...
        marker_color(mod(i, line_types_cnt)+1),'LineWidth' , line_width, 'MarkerSize', marker_size);
    hold on
end
grid
xlabel('Iteration');
ylabel('Relative Error');
legend('Random block NOL','Cyclic block NOL','Random block OL','Cyclic block OL')
set(gca,'FontSize', 17);
hold off
