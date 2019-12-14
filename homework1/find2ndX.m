function x = find2ndX(oriMatrix)
    %To extract the second largest number's position --x 
    %in TrainingSample_BG/FG
    %First, set the largest element in BG/FG matrices to 0
    oriMatrix(oriMatrix == max(oriMatrix,[],2))=0;
    %second, find the largest element in TSDCT_FG/BG, send their positions to
    %xFG/BG.
    [~, x] = max(oriMatrix,[],2);
end