function[sumvalue] = sumfunction(matrix1)
temp = ones(size(matrix1,2),size(matrix1,1));
if size(matrix1,1)>size(matrix1,2)
    sumvalue = temp*matrix1;
else
    sumvalue = matrix1*temp;
end