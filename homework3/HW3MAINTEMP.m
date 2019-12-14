function[u_finalFG, Covfinal_FG, u_finalBG, Covfinal_BG] = HW3MAINTEMP(D1_FG, D1_BG, W1, u0_FG, u0_BG, alpha, counter, MAP, MLE)
% The ECE271A HW3 Bayes Parameter Estimate
% author: Haoyang Ding PID:A53320920
% email: had035@eng.ucsd.edu
%-------------------------------------------------------------------------------------------

%First P(x|u)'s covarience
%Cov_FG Cov_BG is the covarience of P(x|u) P(x|u)~G(u,Cov)
Cov_FG=covfunction(D1_FG,min(size(D1_FG)));
Cov_BG=covfunction(D1_BG,min(size(D1_BG)));

Cov0=zeros(64,64);
for i=1:64
    Cov0(i,i)=alpha(counter)*W1(i);
end

meanmatrix_FG=zeros(1,min(size(D1_FG)));
meanmatrix_BG=zeros(1,min(size(D1_BG)));
for i=1:min(size(D1_FG))
    meanmatrix_FG(i)=mean(D1_FG(:,i));
    meanmatrix_BG(i)=mean(D1_BG(:,i));
end

% P(u|D)~G(u_n,Cov_n) u_n is u_1 here for D1
u_nFG=Cov0 / (Cov0+Cov_FG/max(size(D1_FG))) * meanmatrix_FG'+ (1/max(size(D1_FG))) * Cov_FG / (Cov0+Cov_FG/max(size(D1_FG))) * u0_FG';
u_nBG=Cov0 / (Cov0+Cov_BG/max(size(D1_BG))) * meanmatrix_BG' + (1/max(size(D1_BG))) * Cov_BG / (Cov0+Cov_BG/max(size(D1_BG))) * u0_BG';

% Cov_nFG Cov_nBG is the Cov_n  
Cov_nFG = Cov0 / (Cov0+Cov_FG/max(size(D1_FG))) * Cov_FG/max(size(D1_FG));
Cov_nBG = Cov0 / (Cov0+Cov_BG/max(size(D1_BG))) * Cov_BG/max(size(D1_BG));

% p(x|D)~G(u_n,Cov+Cov_n)=G(u_nFG/BG,Cov_FG/BG+Cov_nFG/BG)
u_finalFG=u_nFG;
u_finalBG=u_nBG;
Covfinal_FG=Cov_FG+Cov_nFG;
Covfinal_BG=Cov_BG+Cov_nBG;
if MAP==1
    %Covfinal_FG=Cov_FG*( 1+1/max( size(D1_FG) ) );
    %Covfinal_BG=Cov_BG*( 1+1/max( size(D1_BG) ) );
    Covfinal_FG = Cov_FG;
    Covfinal_BG = Cov_BG;
end

if MLE==1
    u_finalFG=meanmatrix_FG.';
    u_finalBG=meanmatrix_BG.';
    Covfinal_FG=Cov_FG;
    Covfinal_BG=Cov_BG;
end