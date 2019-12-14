function[gaussian] = multigaussian(matrix,mean,cov)
temp1=(2*pi)^(-max(size(matrix))/2);
temp2=det(cov);
temp2=temp2^(-1/2);
temp3=(matrix-mean).' * inv(cov) * (matrix-mean);

gaussian = temp1 * temp2 * exp(-temp3/2);
end