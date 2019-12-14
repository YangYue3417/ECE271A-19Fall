%set an empty workspace
clc;clear;

%load the Traning Samples
[BG, FG] = loadTS_DCT8('-mat','TrainingSamplesDCT_8_new.mat');

%Print the histogram
figure('Name','histogram of tranning samples');
histo = histogram([zeros([1,size(BG,1)]),ones([1,size(FG,1)])],'Normalization','Probability');
title('histogram of tranning samples')

%Calculate the Prior Probablity of {cheetah, grass}
Py_cheetah = size(FG,1)/(size(FG,1)+size(BG,1));
Py_grass = 1 - Py_cheetah;

%Calculate the ML Properties
%[BGmu1,BGsigma1] = GaussianCalc(BG);
[BGmu,BGsigma] = MLpropsCalc(BG);
%[FGmu1,FGsigma1] = GaussianCalc(FG);
[FGmu,FGsigma] = MLpropsCalc(FG);

%plot the gaussian distribution
figcnt = 1;
figure(figcnt);
%suptitle('Gaussian Distribution of x = 1,...,16');
for i = 1:16
    GaussPlot(BGmu(i),BGsigma(i),FGmu(i),FGsigma(i),i);
end
figcnt = figcnt+1;
figure(figcnt);
%suptitle('Gaussian Distribution of x = 17,...,32');
for i = 17:32
    GaussPlot(BGmu(i),BGsigma(i),FGmu(i),FGsigma(i),i);
end
figcnt = figcnt+1;
figure(figcnt);
%suptitle('Gaussian Distribution of x = 33,...,48');
for i = 33:48
    GaussPlot(BGmu(i),BGsigma(i),FGmu(i),FGsigma(i),i);
end
figcnt = figcnt+1;
figure(figcnt);
%suptitle('Gaussian Distribution of x = 49,...,64');
for i = 49:64
    GaussPlot(BGmu(i),BGsigma(i),FGmu(i),FGsigma(i),i);
end
figcnt = figcnt+1;
%From the distribution plots can we imply that X = 1 case may be the most
%distinguishable

% pick 8 featrues out of 64 features and plot
% #1,#12,#14,#18,#19,#25,#32,#33
figure('Name','Gaussian Distribution of 8 best features');
best8array = [1,12,14,18,19,25,32,33];
for i = 1:8
    subplot(4,2,i);
    mu1 = BGmu(best8array(i));
    mu2 = FGmu(best8array(i));
    sig1 = BGsigma(best8array(i));
    sig2 = FGsigma(best8array(i));
    xmin = min(mu1,mu2)-2*(sig1+sig2);
    xmax = max(mu1,mu2)+2*(sig1+sig2);
    x = xmin:(xmax-xmin)/100:xmax;
    y1 = (sqrt(2*pi)*sig1).^(-1) * exp(-(x-mu1).^2/(2*sig1*sig1));
    y2 = (sqrt(2*pi)*sig2).^(-1) * exp(-(x-mu2).^2/(2*sig2*sig2));
    plot(x,y1);
    hold on
    plot(x,y2);
    title(best8array(i));
end

% pick 8 featrues out of 64 features and plot
% #2,#3,#4,#5,#59,#60,#63,#64
hold off
figure('Name','Gaussian Distribution of 8 worst features');
worst8array = [2,3,4,5,59,60,63,64];
for i = 1:8
    subplot(4,2,i);
    mu1 = BGmu(worst8array(i));
    mu2 = FGmu(worst8array(i));
    sig1 = BGsigma(worst8array(i));
    sig2 = FGsigma(worst8array(i));
    xmin = min(mu1,mu2)-2*(sig1+sig2);
    xmax = max(mu1,mu2)+2*(sig1+sig2);
    x = xmin:(xmax-xmin)/100:xmax;
    y1 = (sqrt(2*pi)*sig1).^(-1) * exp(-(x-mu1).^2/(2*sig1*sig1));
    y2 = (sqrt(2*pi)*sig2).^(-1) * exp(-(x-mu2).^2/(2*sig2*sig2));
    plot(x,y1);
    hold on
    plot(x,y2);
    title(worst8array(i));
end

%Load the Cheetah.jpg, using DCT and zigzag method
%read the test img cheeta
cheetah_img = imread('cheetah.bmp');
cheetah_dw = im2double(cheetah_img);

%set a blank padding
cheetah_pad = [cheetah_dw, zeros([size(cheetah_dw,1),7]); zeros([7,size(cheetah_dw,2)+7])];

%initialize test set
img_uzz = zeros([255,270,64]);

%test set implementation
%slice the image into 8*8 blocks and run the dct2()
for col = (1:size(cheetah_dw,2))
    if (col+7) > size(cheetah_pad,2)
        break;
    end
    for row = (1:size(cheetah_dw,1))
        if (row+7) > size(cheetah_pad,1)
            break;
        end
        temp = reshape(dct2(cheetah_pad(row:row+7,col:col+7)),1,[]);
        img_uzz(row,col,:) = temp;
    end
end

img_uzzR = reshape(img_uzz,[],64);

%load and use zigzag to search the X=? position
zigzag = load('Zig-Zag Pattern.txt');
zigzag = reshape(zigzag, 1, []) + 1;

% the chosen Property in the training set Xtrain -> the chosen Property in
% the testing set Xtest 
% (zigzag-ed -> unzigzag-ed)
img_zz = zeros(size(img_uzzR));
for i = 1:size(zigzag,2)
    img_zz(:,zigzag(i)) = img_uzzR(:,i);
end

% Use BDR to decide -64d
img64 = zeros(size(cheetah_dw));
convBG = (BG-BGmu).'*(BG-BGmu)/size(BG,1);
convFG = (FG-FGmu).'*(FG-FGmu)/size(FG,1);

for i = 1:size(img_zz,1)
    bdrBG = ndGauss(img_zz(i,:),BGmu,convBG)*Py_grass;
    bdrFG = ndGauss(img_zz(i,:),FGmu,convFG)*Py_cheetah;
    if (bdrBG/bdrFG)>= 1
        img64(i) = 0;
    else
        img64(i) = 1;
    end
end

img64 = reshape(img64,size(cheetah_dw,1),size(cheetah_dw,2));
% img64(size(img64,1)-7:size(img64,1),:) = [];
% img64(:,size(img64,2)-7:size(img64,2)) = [];

figure('Name','Cheetah with 64 features');
imagesc(img64);
colormap(gray(255));

% Use BDR to decide -64d
img8 = zeros(size(cheetah_dw));
convBG8 = (BG(:,best8array)-BGmu(:,best8array)).'*(BG(:,best8array)-BGmu(:,best8array))/size(BG,1);
convFG8 = (FG(:,best8array)-FGmu(:,best8array)).'*(FG(:,best8array)-FGmu(:,best8array))/size(FG,1);

for i = 1:size(img_zz,1)
    bdrBG = ndGauss(img_zz(i,best8array),BGmu(best8array),convBG8)*Py_grass;
    bdrFG = ndGauss(img_zz(i,best8array),FGmu(best8array),convFG8)*Py_cheetah;
    if (bdrBG/bdrFG)>= 1
        img8(i) = 0;
    else
        img8(i) = 1;
    end
end

img8 = reshape(img8,size(cheetah_dw,1),size(cheetah_dw,2));
% img8(size(img8,1)-7:size(img8,1),:) = [];
% img8(:,size(img8,2)-7:size(img8,2)) = [];

figure('Name','Cheetah with 8 best features');
imagesc(img8);
colormap(gray(255));

% accuracy calculate
cheetah_mask = imread('cheetah_mask.bmp');
cheetah_maskdw = im2double(cheetah_mask);

% err_num64 = sum_y(sum_y(abs(cheetah_maskdw(1:247,1:262)-img64),1),2);
err_num64 = sum_y(sum_y(abs(cheetah_maskdw-img64),1),2);
total_num = size(cheetah_maskdw,1)*size(cheetah_maskdw,2);
err_rate64 = err_num64/total_num;
fprintf('\nError ratio by using 64 properties for Gaussian: %f\n',err_rate64);

% err_num8 = sum_y(sum_y(abs(cheetah_maskdw(1:247,1:262)-img8),1),2);
err_num8 = sum_y(sum_y(abs(cheetah_maskdw-img8),1),2);
err_rate8 = err_num8/total_num;
fprintf('\nError ratio by using 8 best properties for Gaussian: %f\n',err_rate8);








