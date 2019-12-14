function [cheetah_pred] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
%Compute to distinguish the cheetah
%   Author:Yang Yue
%   PID:A53301503
%   email:y5yue@ucsd.edu
%   ----------------------
nBG = size(D_BG,1);
nFG = size(D_FG,1);

%mu of different datasets -- mu_ML
muD_BG = ysum(D_BG,1)/nBG;
muD_FG = ysum(D_FG,1)/nFG;

%covariances of different datasets  -- cov
meanMatD_BG = zeros(64,64);
meanMatD_FG = zeros(64,64);

for i = 1:nBG
    meanMatD_BG(i,:) = D_BG(i,:)-muD_BG;
end
for i = 1:nFG
    meanMatD_FG(i,:) = D_FG(i,:)-muD_FG;
end

covD_BG = meanMatD_BG.'*meanMatD_BG/nBG;
covD_FG = meanMatD_FG.'*meanMatD_FG/nFG;

% P_mu|T(mu|D)=G(mu,muBG_n,covBG_n) 
cov0 = diag(alpha*W0);
muBG0 = mu0_BG;
muBG_n = nBG*cov0/(covD_BG+nBG*cov0)*muD_BG' + covD_BG/(covD_BG+nBG*cov0)*muBG0';
covBG_n = (covD_BG*cov0)/(covD_BG+nBG*cov0);

% P_mu|T(mu|D)=G(mu,muFG_n,covFG_n) 
muFG0 = mu0_FG; % 1*64
muFG_n = nFG*cov0/(covD_FG+nFG*cov0)*muD_FG' + covD_FG/(covD_FG+nFG*cov0)*muFG0';
covFG_n = (covD_FG*cov0)/(covD_FG+nFG*cov0);

covBG = covD_BG+covBG_n;
covFG = covD_FG+covFG_n;

if str == 0 %predictive
    BG_mu = muBG_n;
    BG_cov = covBG;

    FG_mu = muFG_n;
    FG_cov = covFG;
end
if str == 1 %MAP
    BG_mu = muBG_n;
    BG_cov = covD_BG;

    FG_mu = muFG_n;
    FG_cov = covD_FG;
end
if str == 2 %MLE
    BG_mu = muD_BG';
    BG_cov = covD_BG;

    FG_mu = muD_FG';
    FG_cov = covD_FG;
end

cheetah_pred = zeros(size(cheetah_blocks,1),1);
for xi=1:size(cheetah_blocks,1)
    pFG = ndGauss(cheetah_blocks(xi,:),FG_mu,FG_cov);
    pBG = ndGauss(cheetah_blocks(xi,:),BG_mu,BG_cov);
    if (pFG/pBG)>1
        cheetah_pred(xi)=1;
    else
        continue
    end
end

end

