%% a)
figure('Name','PoE of Mixture Gaussian (C=8) with EM')
str_legend = [];
for i = 1:5
    for j = 1:5
        plot(dim,poe_1{i,j});
        xlabel('Dimension')
        xticks(dim)
        ylabel('PoE')
        str_legend = [str_legend,'BG',num2str(i),'-FG',num2str(j)];
        hold on
    end
end
xticks(dim)
title('a)-PoE of Mixture Gaussian (C=8) with EM')
legend('BG_1 - FG_1','BG_1 - FG_2','BG_1 - FG_3','BG_1 - FG_4','BG_1 - FG_5',...
       'BG_2 - FG_1','BG_2 - FG_2','BG_2 - FG_3','BG_2 - FG_4','BG_2 - FG_5',...
       'BG_3 - FG_1','BG_3 - FG_2','BG_3 - FG_3','BG_3 - FG_4','BG_3 - FG_5',...
       'BG_4 - FG_1','BG_4 - FG_2','BG_4 - FG_3','BG_4 - FG_4','BG_4 - FG_5',...
       'BG_5 - FG_1','BG_5 - FG_2','BG_5 - FG_3','BG_5 - FG_4','BG_5 - FG_5');
grid()
%% b)
figure('Name','PoE of Mixture Gaussian (various C) with EM')
for i = 1:6
    plot(dim,poe_2(i,:))
    xlabel('Dimension')
    ylabel('PoE')
    hold on
end
legend('C = 1','C = 2','C = 4','C = 8','C = 16','C = 32');
xticks(dim)
grid()
title('b)-PoE of Mixture Gaussian (various C) with EM')
