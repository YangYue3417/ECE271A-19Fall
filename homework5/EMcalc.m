function [pi_fin,mu_fin,cov_fin] = EMcalc(dim,C,xi,pi_init,mu_init,cov_init,num)
% initialize for the first round iteration
pi = pi_init;
mu = mu_init;
cov = cov_init;
h = zeros(size(xi,1),C);

for i_iter = 1:1000
    for j = 1:C
        h(:,j) = mvnpdf(xi,mu(j,:),diag(cov(j,:)))*pi(j);
    end
    h = h./repmat(sum(h,2),1,C);
%     save(['save_h_data/',num2str(num),'_iter_',num2str(i_iter),'.mat'],'h');
    sum_h = sum(h,1);
    pi_next = sum_h'/size(xi,1);
    for j = 1:C
        mu_next(j,:) = sum(repmat(h(:,j),1,dim).*xi,1)/sum_h(j);
        cov_next(j,:) = sum(repmat(h(:,j),1,dim).*(xi-repmat(mu(j,:),size(xi,1),1)).^2,1)/sum_h(j);
        cov_next(cov_next<0.0001) = 0.0001;
    end
    pi = pi_next;
    mu = mu_next;
    cov = cov_next;
end
pi_fin = pi;
mu_fin = mu;
cov_fin = cov;
end