function out = LSPK_U(A,Y,U,para)
% Log-sum reg. Kaczmarz based high-order tensor recovery 
% 
% Inputs:
% Y: n1 x n2 x n3 ... x nm
% A: n1 x k x n3 ...x nm
% U: linear transform matrices
%
% para.alpha-step size for coordinate descent
%     .lambda-soft thresholding parameter
%     .maxit-maximum number of iterations
%     .bs-batch size
%     .control: 'rand', 'cyc'
%     .eps-epsilon
%     .controltype: 'block', 'batch'
%     .type: 'sparse', 'lowrank'
% Output:
% X: recovered tensor of size k x n2 x n3 x...xnm
%
% Written by Katherine Henneberger


ndim = length(size(Y));
nway = zeros(1,ndim);
for i = 1:ndim
    nway(i) = size(Y,i);
end

lambda = para.lambda;
alpha  = para.alpha;
bs     = para.bs;
eps    = para.eps;
nb = floor(nway(1)/bs); % number of blocks
display = false; % print iter number
numblock = para.block;
edges = round(linspace(1,nway(1)+1,numblock+1));
idx = randperm(nway(1));

k = size(A,2);
Xsize = nway;
Xsize(1) = k;
X = zeros(Xsize);


Z = X;
obj = zeros(para.maxit,1);
if isfield(para,'gth')
    err = zeros(para.maxit,1);
end

for i = 1:para.maxit
    if ~mod(i,10) && display
        fprintf('iter = %d\n',i);
    end
    switch para.controltype
        case 'batch'
            switch para.control
                case 'rand'
                    % randomized batch
                    ii = ceil(rand*nb);
                    out.ii = ii;
                    ik = idx((bs*(ii-1)+1):(bs*ii));
                case 'cyc'
                    % cyclic batch
                    ii = i;
                    ik = mod((bs*(ii-1)+1):(bs*ii),nway(1));
                    ik(ik==0) = nway(1);
            end
        case 'block'
            switch para.control
                case 'rand'
                    block_select = randi(numblock);
                    ik           = edges(block_select):edges(block_select+1)-1;
                case 'cyc'
                  
                        block_select =  i;
                        block_select = mod(block_select,numblock);
                        if block_select ==0
                            block_select = 3;
                        end
                        %disp(block_select);
                        ik           = edges(block_select):edges(block_select+1)-1;
            
            end
    end

   % normA = norm(A(ik,:,:,:),'fro')^2;
    normA = sum(A(ik,:,:,:).^2,"all");
    
    % coordinate descent
    Z = Z + alpha*htprod_U(htran(A(ik,:,:,:),U),(Y(ik,:,:,:)-htprod_U(A(ik,:,:,:),X,U))./normA,U); 
    
    out.Z = Z;
    % proximal operator
    if sum(sum(sum(isnan(Z))))>1
        break
    end 
    
    switch para.type
        case 'sparse'
            X = prox_lsp(Z,lambda,eps);
            
        case 'lowrank'
            X = prox_lsp_sing(Z,lambda,eps);
    end
    
    if isfield(para,'gth') % relative error
        err(i) = norm(para.gth-X,'fro')./norm(para.gth,'fro');
    end
    % store objective function
    logsum = 0;
    for i1 = 1:Xsize(1)
        for i2 = 1:Xsize(2)
            for i3 = 1:Xsize(3)
                for i4 = 1:Xsize(4)
                    logsum =logsum+(1+(X(i1,i2,i3,i4)/eps)); 
                end 
            end 
        end 
    end
    obj(i) = (1/2)*norm(X,'fro')^2 + lambda*logsum;
    
    
end

out.finaliter = i;
out.X = X;
out.obj = obj;
if isfield(para,'gth')
    out.err = err(1:i);
end

