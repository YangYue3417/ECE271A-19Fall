function [mu, sigma] = MLpropsCalc(samples)
    mu = sum_y(samples,2)/size(samples,1);
    sigma_2 = sum_y((samples-mu).*(samples-mu),2)./size(samples,1);
    sigma = sqrt(sigma_2);
end