function g = ndGauss(x,mu,conv)
    if ~istall(x)
        x = x.';
    end
    if ~istall(mu)
        mu = mu.';
    end
    g = (2*pi)^(-size(x,1)/2)*(det(conv)^(-1/2))*exp(-((x-mu).')/(conv)*(x-mu)*1/2);
end