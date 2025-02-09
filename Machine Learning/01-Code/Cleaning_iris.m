clear; close all;
Features=xlsread('iris.xlsx','iris','A1:D150');
[~,Y]=xlsread('iris.xlsx','iris','E1:E150');
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
id=35;
index1=find( Features(:,1)==Features(id,1));
index2=find( Features(:,2)==Features(id,2));
index3=find( Features(:,3)==Features(id,3));
index4=find( Features(:,4)==Features(id,4));
index=intersect(index1,intersect(index2,intersect(index3,index4)))
Features(index,:)
Y(index)