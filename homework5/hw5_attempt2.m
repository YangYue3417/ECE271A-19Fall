%% Step1:initialization, calculate DCT and do the zigzag transformation
clc;clear;
% Once again we use the decomposition
% into 8 ¡Á 8 image blocks, compute the DCT of each block, and zig-zag scan.
zigzag = load('Zig-Zag Pattern.txt');
zigzag = reshape(zigzag, 1, []) + 1;

cheetah_img = imread('cheetah.bmp');
cheetah_dw = im2double(cheetah_img);

cheetah_mask = imread('cheetah_mask.bmp');
cheetah_maskdw = im2double(cheetah_mask);

%set a blank padding
% cheetah_pad = [cheetah_dw, zeros([size(cheetah_dw,1),7]); zeros([7,size(cheetah_dw,2)+7])];
[img_row, img_col] = size(cheetah_dw);
% cheetah_blocks = zeros(img_row*img_col,64);
cheetah_blocks = zeros((img_row-7)*(img_col-7),64);
cnt = 1;

for col = 1:img_col-7
    for row = 1:img_row-7
%         window = cheetah_pad(row:row+7,col:col+7);
        window = cheetah_dw(row:row+7,col:col+7);
        cheetah_blocks(cnt,:) = reshape(dct2(window),[],64);
        cnt = cnt+1;
    end
end
cheetah_blocks = dozigzag(cheetah_blocks,zigzag);
clear cheetah_img cheetah_mask cnt window cheetah_pad zigzag row col
%% 1 - initialization
load('TrainingSamplesDCT_8_new.mat');
priBG = size(TrainsampleDCT_BG,1)/(size(TrainsampleDCT_BG,1)+size(TrainsampleDCT_FG,1));
priFG = 1-priBG;
dim = [1,2,4,8,16,24,32,40,48,56,64];
times_1 = 5;
%% Step-1 BG
C1 = 8;
muBG = cell(times_1,size(C1,2));
covBG = cell(times_1,size(C1,2));
piBG = cell(times_1,size(C1,2));
count = 1;
for dim_i = 64
    for time_i = 1:times_1
        % Initialize the BG's pre-requisites
        xi_BG = TrainsampleDCT_BG(:,1:dim_i);
        [pi_init_BG,mu_init_BG,cov_init_BG] = randomInit(dim_i,C1);
        [pi,mu,cov] = EMcalc(dim_i,C1,xi_BG,pi_init_BG,mu_init_BG,cov_init_BG,1);

        muBG{time_i,count} = mu;
        covBG{time_i,count} = cov;
        piBG{time_i,count} = pi; 
    end
    count = count + 1;
end
clear mu pi cov count dim_i xi_BG pi_init_BG mu_init_BG cov_init_BG time_i
%% Step-1 FG
muFG = cell(times_1,size(C1,2));
covFG = cell(times_1,size(C1,2));
piFG = cell(times_1,size(C1,2));
count = 1;
for dim_i = 64
    for time_i = 1:times_1
        % Initialize the FG's pre-requisites
        xi_FG = TrainsampleDCT_FG(:,1:dim_i);
        [pi_init_FG,mu_init_FG,cov_init_FG] = randomInit(dim_i,C1);
        [pi,mu,cov] = EMcalc(dim_i,C1,xi_FG,pi_init_FG,mu_init_FG,cov_init_FG,2);

        muFG{time_i,count} = mu;
        covFG{time_i,count} = cov;
        piFG{time_i,count} = pi; 
    end
    count = count + 1;
end
clear mu pi cov count dim_i xi_FG pi_init_FG mu_init_FG cov_init_FG time_i
%%
poe_1 = cell(5);
for dim_i = 1:size(dim,2)
    test_set = cheetah_blocks(:,1:dim(dim_i));
    for i = 1:times_1
        Px_BG = cheetah_BDR(test_set,C1,dim(dim_i),muBG{i},covBG{i},piBG{i},priBG);
        for j = 1:times_1
            Px_FG = cheetah_BDR(test_set,C1,dim(dim_i),muFG{j},covFG{j},piFG{j},priFG);
            cheetah_vec = Px_FG./Px_BG;
            cheetah_vec(cheetah_vec>1) = 1;
            cheetah_vec(cheetah_vec~=1) = 0;
            cheetah_res = reshape(cheetah_vec,img_row-7,img_col-7);
            cheetah_res_pad = [cheetah_res,zeros([size(cheetah_res,1),7]); zeros([7,size(cheetah_res,2)+7])];
            poe_1{i,j}(dim_i) = sum(abs(cheetah_maskdw-cheetah_res_pad),'all')/img_row/img_col;
        end
    end
end
clear i j Px_BG Px_FG cheetah_vec
%%
figure
imagesc(cheetah_res_pad)
colormap('gray')

%% Step-2 BG
times_2 = 1;
C2 = [1,2,4,8,16,32];

%%
muBG = cell(times_2,size(C2,2));
covBG = cell(times_2,size(C2,2));
piBG = cell(times_2,size(C2,2));
count = 1;
for dim_i = 64
    for time_i = 1:times_2
        for c_i = C2
            % Initialize the BG's pre-requisites
            xi_BG = TrainsampleDCT_BG(:,1:dim_i);
            [pi_init_BG,mu_init_BG,cov_init_BG] = randomInit(dim_i,c_i);
            [pi,mu,cov] = EMcalc(dim_i,c_i,xi_BG,pi_init_BG,mu_init_BG,cov_init_BG,1);

            muBG{time_i,count} = mu;
            covBG{time_i,count} = cov;
            piBG{time_i,count} = pi; 
            count = count + 1;
        end
    end
end
clear mu pi cov count dim_i xi_BG pi_init_BG mu_init_BG cov_init_BG time_i

%% Step-2 FG
muFG = cell(times_2,size(C2,2));
covFG = cell(times_2,size(C2,2));
piFG = cell(times_2,size(C2,2));
count = 1;
for dim_i = 64
    for time_i = 1:times_2
        for c_i = C2
            % Initialize the FG's pre-requisites
            xi_FG = TrainsampleDCT_FG(:,1:dim_i);
            [pi_init_FG,mu_init_FG,cov_init_FG] = randomInit(dim_i,c_i);
            [pi,mu,cov] = EMcalc(dim_i,c_i,xi_FG,pi_init_FG,mu_init_FG,cov_init_FG,2);

            muFG{time_i,count} = mu;
            covFG{time_i,count} = cov;
            piFG{time_i,count} = pi; 
            count = count + 1;
        end
    end
end
clear mu pi cov count dim_i xi_BG pi_init_BG mu_init_BG cov_init_BG time_i

%% 
poe_2 = zeros(size(C2,2),size(dim,2));
for dim_i = 1:size(dim,2)
    test_set = cheetah_blocks(:,1:dim(dim_i));
    for c_i = 1:size(C2,2)
        for i = 1:times_2
            Px_BG = cheetah_BDR(test_set,C2(c_i),dim(dim_i),muBG{c_i},covBG{c_i},piBG{c_i},priBG);
            Px_FG = cheetah_BDR(test_set,C2(c_i),dim(dim_i),muFG{c_i},covFG{c_i},piFG{c_i},priFG);
            cheetah_vec = Px_FG./Px_BG;
            cheetah_vec(cheetah_vec>1) = 1;
            cheetah_vec(cheetah_vec~=1) = 0;
%                 cheetah_res = reshape(cheetah_vec,img_row,img_col);
            cheetah_res = reshape(cheetah_vec,img_row-7,img_col-7);
            cheetah_res_pad = [cheetah_res,zeros([size(cheetah_res,1),7]); zeros([7,size(cheetah_res,2)+7])];
            poe_2(c_i,dim_i) = sum(abs(cheetah_maskdw-cheetah_res_pad),'all')/img_row/img_col;
        end
    end
end
clear i j Px_BG Px_FG cheetah_vec

%%
figure
imagesc(cheetah_res_pad)
colormap('gray')

%%
figure
plot(dim,poe_2(1,:))

%%
format long
t = datetime('now');
save(['poe_save/poe_1_',datestr(now,30),'.mat'],'poe_1');
save(['poe_save/poe_2_',datestr(now,30),'.mat'],'poe_2');