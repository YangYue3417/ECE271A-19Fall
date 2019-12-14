function a = sum_y(matrix, dim)
    if dim == 1
        a = matrix*ones([size(matrix,2),1]);
    elseif dim == 2
        a = ones([1,size(matrix,1)])*matrix;
    end    
end