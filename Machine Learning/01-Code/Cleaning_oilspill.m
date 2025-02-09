clear; close all;
Data=xlsread('oil-spill.xlsx','oil-spill','A1:AX937');
Y=Data(:,end);
Features=Data(:,2:end-1);
size(Features)
clear Data
disp('Zero Variance')
index=find(var(Features)==0)
disp('Variance smaller than 0.1')
index=find(var(Features)<0.1)
disp('Unique values')
[Featun,index]=unique(Features,'rows','stable');
size(Featun)
Yun=Y(index);
disp('Outlier?')
figure
plot(Features(:,1),'*')
figure
plot(Features(:,2),'*')