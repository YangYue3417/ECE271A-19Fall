function [mu, sigma_2] = GaussianCalc(samples)
    mu = mean(samples,1);
    sigma_2 = samples(1,:)
end