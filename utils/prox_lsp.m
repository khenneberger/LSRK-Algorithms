function X = prox_lsp(Z, lambda, eps)


p = length(size(Z));
n = zeros(1,p);
for i = 1:p
    n(i) = size(Z,i);
end

X = zeros(n);
% L = ones(1,p);
% for i = 3:p
%     Z = fft(Z,[],i);
%     L(i) = L(i-1) * n(i);
% end

for i1 = 1:n(1)
    for i2 = 1:n(2)
        for i3 = 1:n(3)
            for i4 = 1:n(4)
                
                
                if sqrt(lambda)<=eps
                    if abs(Z(i1,i2,i3,i4)) <= lambda/eps
                        X(i1,i2,i3,i4) = 0;
                    else
                        X(i1,i2,i3,i4) = sign(Z(i1,i2,i3,i4))*r_eval(abs(Z(i1,i2,i3,i4)),lambda,eps);
                    end
                else
                    func = @(x) root_func(x,lambda,eps);
                    root = fzero(func,[2*sqrt(lambda)-eps,lambda/eps]);
                    %disp(root);
                    if abs(Z(i1,i2,i3,i4))<=root
                        X(i1,i2,i3,i4) = 0;
                    else 
                        X(i1,i2,i3,i4) = sign(Z(i1,i2,i3,i4))*r_eval(abs(Z(i1,i2,i3,i4)),lambda,eps);
                    end
                    % if abs(Z(i1,i2,i3,i4))==root
                    %     X(i1,i2,i3,i4) = ;
                    % end
                end
            end
        end
    end
end

% for i = p:-1:3
%     X = (ifft(X,[],i));
% end
% %disp(r);
% X = real(X);
end
%% 
function r2 = r_eval(x,lambda,eps)
    r2 = (0.5*(x-eps)+sqrt(0.25*(x+eps)^2-lambda));
    %disp(sqrt(0.25*(x+eps)^2-lambda));
end
function root = root_func(x,lambda,eps)
    root = 0.5*(r_eval(x,lambda,eps)-x)^2+lambda*log(1+(abs(r_eval(x,lambda,eps))/eps))-0.5*(x)^2;
end