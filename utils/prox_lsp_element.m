function X = prox_lsp_element(Z, lambda, eps)

                
                if sqrt(lambda)<=eps
                    if abs(Z) <= lambda/eps
                        X = 0;
                    else
                        X = sign(Z)*r_eval(abs(Z),lambda,eps);
                    end
                else
                    func = @(x) root_func(x,lambda,eps);
                    root = fzero(func,[2*sqrt(lambda)-eps,lambda/eps]);
                    %disp(root);
                    if abs(Z)<=root
                        X = 0;
                    else 
                        X = sign(Z)*r_eval(abs(Z),lambda,eps);
                    end
                    % if abs(Z(i1,i2,i3,i4))==root
                    %     X(i1,i2,i3,i4) = ;
                    % end
                end
end     
%% 
function r2 = r_eval(x,lambda,eps)
    r2 = (0.5*(x-eps)+sqrt(0.25*(x+eps)^2-lambda));
    %disp(sqrt(0.25*(x+eps)^2-lambda));
end
function root = root_func(x,lambda,eps)
    root = 0.5*(r_eval(x,lambda,eps)-x)^2+lambda*log(1+(abs(r_eval(x,lambda,eps))/eps))-0.5*(x)^2;
end