function [pri_next,mu_next,cov_next] = mStepCalc(x,mu,cov,h,C)
    % compute the M-step, parameters
    for j = 1:C
        sumh = sum(h(:, j));
        mu_next(j, :) = h(:, j)'*x(:, :)./sumh;
        picur(j) = sumh / row;
    end
    
    for j=1:C
       sum_1=0;
       sum_2=0;
       for i=1: size(dataset,1)
           sum_1=sum_1+h(i,j)*(dataset(i,:)-mucur(j,:)).^2;
           sum_2=sum_2+h(i,j);
       end
       varcur(j, :) = sum_1 / sum_2;
       varcur(varcur < 0.002) = 0.002;
   end
end