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

%% initialization of mu, sigma and pi
load('TrainingSamplesDCT_8_new.mat');

prior_BG = cell(1,5);
mu_BG = cell(1,5);
cov_BG = cell(1,5);


prior_FG = cell(1,5);
mu_FG = cell(1,5);
cov_FG = cell(1,5);

dim = [1,2,4,8,16,24,32,40,48,56,64];
C1 = 8;

%%
for i_dim = dim
    for time = 1:5
        % random initialization
        [pri_Init,mu_Init,cov_Init] = randomInit(dim,C1);
        BG_train = TrainsampleDCT_BG(:,1:i_dim);
        pri = pri_Init;
        mu = mu_Init;
        cov = cov_Init;
        Q_max = 0;
        % iteration of parameter
        while (1)
            [h] = eStepCalc(BG_train,pri,mu,cov,C1);
            [Q] = qCalc(BG_train, h);
            if Q_max >= Q
                prior_BG{time} = pri;
                mu_BG{time} = mu;
                cov_BG{time} = cov;
                break
            else
                Q = Q_max;
                [pri, mu, cov] = mStepCalc(BG_train, h, pri, mu, cov, C1);
            end
        end
        
        % random initialization
        [pri_Init,mu_Init,cov_Init] = randomInit(dim,C1);
        FG_train = TrainsampleDCT_FG(:,1:i_dim);
        pri = pri_Init;
        mu = mu_Init;
        cov = cov_Init;
        Q_max = 0;
        % parameter iteration
        while (1)
            [h] = eStepCalc(BG_train,pri,mu,cov,C1);
            [Q] = qCalc(BG_train, h);
            if Q_max >= Q
                prior_FG{time} = pri;
                mu_FG{time} = mu;
                cov_FG{time} = cov;
                break
            else
                Q = Q_max;
                [pri, mu, cov] = mStepCalc(FG_train, h, pri, mu, cov, C1);
            end
        end
    end
end

%%
BG_train = TrainsampleDCT_BG(:,1:2);
pri = pri_Init;
mu = mu_Init;
cov = cov_Init;
[h] = eStepCalc(BG_train,pri,mu,cov,8);