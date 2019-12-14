function[covmatrix] = covfunction(matrix,num)
meanmatrix=zeros(1,num);
for i=1:num
    meanmatrix(i)=sum_y(matrix(:,i),max(size(matrix)))/size(matrix,size(max(size(matrix))));
end
covmatrix=zeros(num,num);

for i=1:num
    for j=1:num
        covmatrix(i,j)=sum((matrix(:,i) - meanmatrix(i)).*(matrix(:,j) - meanmatrix(j)) ) / (max(size(matrix)));
    end
end
end