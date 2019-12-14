function g = ndGauss(x,mu,cov)
    if istall(x)==0
        x = x.';
    end
    if size(x)~=size(mu)
        disp('Size unequal');
    end
    g1 = (2*pi)^(-size(x,1)/2)*(det(cov)^(-1/2))*exp(-((x-mu).')/(cov)*(x-mu)*1/2);
    g=g1;
end