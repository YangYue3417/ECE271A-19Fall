function [prior_next, mu_next, cov_next] = eStepCalc(x, pri, mu, cov, C)
    % compute the hij first E-step
    for i = 1:size(x,1)
        for j = 1:C
            %mvnpdf -- Multivariate normal probability density function
            h(i, j) = mvnpdf(x(i, :), mu(j, :), diag(cov(j, :))) * pri(j);
        end
        h(i, :) = h(i, :) / sum(h(i, :));
    end
end