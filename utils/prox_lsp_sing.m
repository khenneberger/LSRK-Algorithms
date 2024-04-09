function X = prox_lsp_sing(Z, lambda, eps)



p = length(size(Z));
n = zeros(1,p);
for i = 1:p
    n(i) = size(Z,i);
end

X = zeros(n);
L = ones(1,p);
for i = 3:p
    Z = fft(Z,[],i);
    L(i) = L(i-1) * n(i);
end

[U,S,V] = svd(Z(:,:,1),'econ');
%disp(max(S,[],'all'));
S = diag(S);

[n1,n2] = size(S);
for i1 = 1:n1
    for i2 = 1:n2
        S(i1,i2) = prox_lsp_element(S(i1,i2),lambda,eps) ;
    end
end
X(:,:,1) = U*diag(S)*V';

for j = 3 : p
    for i = L(j-1)+1 : L(j)
   %
        I = unfoldi(i,j,L);
        halfnj = floor(n(j)/2)+1;
   %
        if I(j) <= halfnj && I(j) >= 2
            [U,S,V] = svd(Z(:,:,i),'econ');
            S = diag(S);
            [n1,n2] = size(S);
            for i1 = 1:n1
                for i2 = 1:n2
                    S(i1,i2) = prox_lsp_element(S(i1,i2),lambda,eps) ;
                end
            end
                X(:,:,i) = U*diag(S)*V';
                
            
        %Conjugation property
        elseif I(j) > halfnj
            %
            n_ = nc(I,j,n);
            %
            i_ = foldi(n_,j,L);
            X(:,:,i) = conj( X(:,:,i_));
                
        end
    end
end

for i = p:-1:3
    X = (ifft(X,[],i));
end
X = real(X);
