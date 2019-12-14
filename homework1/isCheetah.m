function res = isCheetah(test, prob)
    res = zeros(size(test));
    for i = 1:size(res,1)
        if prob(test(i)) > 1
            res(i) = 1;
        else
            res(i) = 0;
        end
    end
end
