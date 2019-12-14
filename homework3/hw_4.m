%% Step1:initialization, calculate DCT and do the zigzag transformation
clc;
% Once again we use the decomposition
% into 8 ¡Á 8 image blocks, compute the DCT of each block, and zig-zag scan.
zigzag = load('Zig-Zag Pattern.txt');
zigzag = reshape(zigzag, 1, []) + 1;

cheetah_img = imread('cheetah.bmp');
cheetah_dw = im2double(cheetah_img);

cheetah_mask = imread('cheetah_mask.bmp');
cheetah_maskdw = im2double(cheetah_mask);

%set a blank padding
cheetah_pad = [cheetah_dw, zeros([size(cheetah_dw,1),7]); zeros([7,size(cheetah_dw,2)+7])];
[img_row, img_col] = size(cheetah_dw);
cheetah_blocks = zeros(img_row*img_col,64);
cnt = 1;

for col = 1:img_col
    for row = 1:img_row
        window = cheetah_pad(row:row+7,col:col+7);
        cheetah_blocks(cnt,:) = reshape(dct2(window),[],64);
        cnt = cnt+1;
    end
end
cheetah_blocks = dozigzag(cheetah_blocks,zigzag);

load('hw3Data/TrainingSamplesDCT_subsets_8.mat');
%% Strategy 1
load('Alpha.mat');  %
load('Prior_1.mat'); 
% % P_mu|T(mu|D)=G(mu,muBG_n,covBG_n) 
muBG0 = mu0_BG;
% % P_mu|T(mu|D)=G(mu,muFG_n,covFG_n) 
muFG0 = mu0_FG; % 1*64
%% Strategy1-Dataset1
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s1d1_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D1_BDR = cheetahCalc(cheetah_blocks,D1_BG,D1_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS1D1_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d1_BDR(i) = PE_BDR;
end

imgS1D1_MLE = cheetahCalc(cheetah_blocks,D1_BG,D1_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS1D1_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s1d1_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s1d1_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D1_MAP = cheetahCalc(cheetah_blocks,D1_BG,D1_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS1D1_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d1_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy1-Dataset1)')
semilogx(alpha,PE_s1d1_BDR)
grid on
hold on
semilogx(alpha,PE_s1d1_MLE)
hold on 
semilogx(alpha,PE_s1d1_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy1-Dataset1)')

%% S1-Dataset2
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s1d2_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D2_BDR = cheetahCalc(cheetah_blocks,D2_BG,D2_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS1D2_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d2_BDR(i) = PE_BDR;
end

imgS1D2_MLE = cheetahCalc(cheetah_blocks,D2_BG,D2_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS1D2_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s1d2_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s1d2_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D2_MAP = cheetahCalc(cheetah_blocks,D2_BG,D2_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS1D2_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d2_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy1-Dataset2)')
semilogx(alpha,PE_s1d2_BDR)
grid on
hold on
semilogx(alpha,PE_s1d2_MLE)
hold on 
semilogx(alpha,PE_s1d2_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy1-Dataset2)')
%% S1-Dataset3
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s1d3_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D3_BDR = cheetahCalc(cheetah_blocks,D3_BG,D3_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS1D3_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d3_BDR(i) = PE_BDR;
end

imgS1D3_MLE = cheetahCalc(cheetah_blocks,D3_BG,D3_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS1D3_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s1d3_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s1d3_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D3_MAP = cheetahCalc(cheetah_blocks,D3_BG,D3_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS1D3_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d3_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy1-Dataset3)')
semilogx(alpha,PE_s1d3_BDR)
grid on
hold on
semilogx(alpha,PE_s1d3_MLE)
hold on 
semilogx(alpha,PE_s1d3_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy1-Dataset3)')
%% S1-Dataset4
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s1d4_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D4_BDR = cheetahCalc(cheetah_blocks,D4_BG,D4_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS1D4_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d4_BDR(i) = PE_BDR;
end

imgS1D4_MLE = cheetahCalc(cheetah_blocks,D4_BG,D4_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS1D4_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s1d4_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s1d4_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS1D4_MAP = cheetahCalc(cheetah_blocks,D4_BG,D4_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS1D4_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s1d4_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy1-Dataset4)')
semilogx(alpha,PE_s1d4_BDR)
grid on
hold on
semilogx(alpha,PE_s1d4_MLE)
hold on 
semilogx(alpha,PE_s1d4_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy1-Dataset4)')

%% Strategy 2
% load('Alpha.mat');  
load('Prior_2.mat'); 
% % P_mu|T(mu|D)=G(mu,muBG_n,covBG_n) 
muBG0 = mu0_BG;
% % P_mu|T(mu|D)=G(mu,muFG_n,covFG_n) 
muFG0 = mu0_FG; % 1*64
%% Strategy2-Dataset1
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s2d1_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D1_BDR = cheetahCalc(cheetah_blocks,D1_BG,D1_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS2D1_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d1_BDR(i) = PE_BDR;
end

imgS2D1_MLE = cheetahCalc(cheetah_blocks,D1_BG,D1_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS2D1_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s2d1_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s2d1_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D1_MAP = cheetahCalc(cheetah_blocks,D1_BG,D1_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS2D1_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d1_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy2-Dataset1)')
semilogx(alpha,PE_s2d1_BDR)
grid on
hold on
semilogx(alpha,PE_s2d1_MLE)
hold on 
semilogx(alpha,PE_s2d1_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy2-Dataset1)')
%% S2-Dataset2
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s2d2_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D2_BDR = cheetahCalc(cheetah_blocks,D2_BG,D2_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS2D2_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d2_BDR(i) = PE_BDR;
end

imgS2D2_MLE = cheetahCalc(cheetah_blocks,D2_BG,D2_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS2D2_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s2d2_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s2d2_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D2_MAP = cheetahCalc(cheetah_blocks,D2_BG,D2_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS2D2_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d2_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy2-Dataset2)')
semilogx(alpha,PE_s2d2_BDR)
grid on
hold on
semilogx(alpha,PE_s2d2_MLE)
hold on 
semilogx(alpha,PE_s2d2_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy2-Dataset2)')
%% S1-Dataset3
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s2d3_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D3_BDR = cheetahCalc(cheetah_blocks,D3_BG,D3_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS2D3_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d3_BDR(i) = PE_BDR;
end

imgS2D3_MLE = cheetahCalc(cheetah_blocks,D3_BG,D3_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS2D3_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s2d3_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s2d3_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D3_MAP = cheetahCalc(cheetah_blocks,D3_BG,D3_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS2D3_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d3_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy2-Dataset3)')
semilogx(alpha,PE_s2d3_BDR)
grid on
hold on
semilogx(alpha,PE_s2d3_MLE)
hold on 
semilogx(alpha,PE_s2d3_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy2-Dataset3)')
%% S1-Dataset4
%function [img64] = cheetahCalc(cheetah_blocks,D_BG,D_FG,mu0_BG,mu0_FG,alpha,W0,str)
PE_s2d4_BDR = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D4_BDR = cheetahCalc(cheetah_blocks,D4_BG,D4_FG,muBG0,muFG0,alpha(i),W0,0);
    img64 = reshape(imgS2D4_BDR,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_BDR = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d4_BDR(i) = PE_BDR;
end

imgS2D4_MLE = cheetahCalc(cheetah_blocks,D4_BG,D4_FG,muBG0,muFG0,alpha(1),W0,2);
img64 = reshape(imgS2D4_MLE,size(cheetah_dw,1),size(cheetah_dw,2));
PE_MLE = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
PE_s2d4_MLE = ones(1,size(alpha,2))*PE_MLE;

PE_s2d4_MAP = zeros(size(alpha));
for i = 1:size(alpha,2)
    imgS2D4_MAP = cheetahCalc(cheetah_blocks,D4_BG,D4_FG,muBG0,muFG0,alpha(i),W0,1);
    img64 = reshape(imgS2D4_MAP,size(cheetah_dw,1),size(cheetah_dw,2));
    PE_MAP = ysum(ysum(abs(cheetah_maskdw-img64),1),2)/(size(cheetah_maskdw,1)*size(cheetah_maskdw,2));
    PE_s2d4_MAP(i) = PE_MAP;
end

% plot  
figure('Name','PoE-log(\alpha) (Strategy2-Dataset4)')
semilogx(alpha,PE_s2d4_BDR)
grid on
hold on
semilogx(alpha,PE_s2d4_MLE)
hold on 
semilogx(alpha,PE_s2d4_MAP)
hold on
legend('Predictive','ML','MAP')
title('PoE vs log(\alpha) (Strategy2-Dataset4)')