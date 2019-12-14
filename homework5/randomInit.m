function [pi_Init, mu_Init, cov_Init] = randomInit(dim, C)
    pi_Init = rand(C,1);
    pi_Init = pi_Init./sum(pi_Init);
    mu_Init = rand(C,dim)-1;
    cov_Init = rand(C,dim)+1;
%     cov_Init(cov_Init<0.0001) = 0.0001;
end