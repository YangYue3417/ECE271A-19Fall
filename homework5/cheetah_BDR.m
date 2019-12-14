function cheetah_vec = cheetah_BDR(blocks,C,dim,mu,cov,pi,pri)
    cheetah_vec = 0;
    for i = 1:C
        cheetah_vec = cheetah_vec + pi(i)*mvnpdf(blocks,repmat(mu(i,1:dim),size(blocks,1),1),diag(cov(i,1:dim)));
    end
    cheetah_vec = cheetah_vec*pri;
end