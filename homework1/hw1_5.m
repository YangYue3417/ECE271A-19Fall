clc;
clear;

%load training samples and zig-zag order
load('TrainingSamplesDCT_8.mat');
TSDCT_FG = TrainsampleDCT_FG;
TSDCT_BG = TrainsampleDCT_BG;
zigzag = load('Zig-Zag Pattern.txt');
zigzag = reshape(zigzag, 1, []) + 1;

%set prior prob P(X), P(Y)
lenFG = size(TrainsampleDCT_FG, 1);
lenBG = size(TrainsampleDCT_BG, 1);

xFG = find2ndX(TSDCT_FG);
xBG = find2ndX(TSDCT_BG);

%print out histographs of BG/FG
figure(1);
h_fg = histogram(xFG,'Normalization','probability');
title('P_{X|Y}(x|cheetah)');
figure(2);
h_bg = histogram(xBG,'Normalization','probability');
title('P_{X|Y}(x|grass)');

%Pxy_BG <= P(X=xi|Y=BG)
%Pxy_FG <= P(X=xi|Y=FG)
Pxy_BG = zeros([64,1]);
Pxy_FG = zeros([64,1]);
tlb_BG = tabulate(xBG);
tlb_FG = tabulate(xFG);
Pxy_BG(1:size(tlb_BG,1)) = tlb_BG(:,3); 
Pxy_FG(1:size(tlb_FG,1)) = tlb_FG(:,3);
Pxy_BG(Pxy_BG == 0) = 0.0001;

py_BG = lenBG/(lenBG+lenFG);
py_FG = 1-py_BG;
PdP = (Pxy_FG.*(py_FG))./(Pxy_BG.*(py_BG)); %decision boundary

%read the test img cheeta
cheetah_img = imread('cheetah.bmp');
cheetah_dw = im2double(cheetah_img);
%set a blank padding
cheetah_pad = [cheetah_dw, zeros([size(cheetah_dw,1),7]); zeros([7,size(cheetah_dw,2)+7])];

%initialize test set
test_set = zeros([255*270,64]);
cnt = 1;
%test set implementation

%slice the image into 8*8 blocks and run the dct2()
for col = (1:size(cheetah_pad,2))
    if (col+7) > size(cheetah_pad,2)
        break;
    end
    for row = (1:size(cheetah_pad,1))
        if (row+7) > size(cheetah_pad,1)
            break;
        end
        test_set(cnt,:) = reshape(abs(dct2(cheetah_pad(row:row+7,col:col+7))),1,[]);
        cnt = cnt + 1;
    end
end

%find the 2nd max value's position and map it with zigzag
testMx2nd = find2ndX(test_set);
test2ndZig = zeros([size(testMx2nd,1),1]);
for i = 1:size(testMx2nd,1)
    test2ndZig(i) = zigzag(testMx2nd(i)); 
end

%Cheetah or not?
A = reshape(isCheetah(test2ndZig,PdP),size(cheetah_dw,1),size(cheetah_dw,2));
figure(3);
imagesc(A);
colormap(gray(255));

%error computation
cheetah_mask = imread("cheetah_mask.bmp");
cheetah_mask_dw = im2double(cheetah_mask);
errorRate = sum(abs(A - cheetah_mask_dw),'all')/(size(cheetah_mask,1)*size(cheetah_mask,2))













