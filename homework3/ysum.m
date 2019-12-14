function a = ysum(matrix, dim)
    if dim == 2
        a = matrix*ones([size(matrix,2),1]);
    elseif dim == 1
        a = ones([1,size(matrix,1)])*matrix;
    end    
end