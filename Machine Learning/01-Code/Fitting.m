clear; close all;
Data=[%Age	Salary
25	135000
55	260000
27	105000
35	220000
60	240000
65	265000
45	270000
40	300000
50	265000
30	105000];
X=Data(:,1);
Y=Data(:,2);
[X,ord]=sort(X);
Y=Y(ord);
%% Linear Regression
Mat=[ones(size(X)), X];
Coeff1=Mat\Y
figure; subplot(1,2,1); hold on;
plot(X,Y,'d')
hold on
plot(X,Coeff1(1)+Coeff1(2)*X,'-')
Error1=norm( Y-Mat*Coeff1)
%% 5th order Fit
Mat=[ones(size(X)), X, X.^2, X.^3, X.^4, X.^5];
Coeff2=Mat\Y
Error2=norm( Y-Mat*Coeff2)
subplot(1,2,2); hold on;
plot(X,Y,'d')
hold on
plot(X,Mat*Coeff2,'-')
figure; hold on
Xnew=linspace(X(1),X(end),100)';
plot(X,Y,'d')
plot(Xnew,Coeff2(1)+Coeff2(2)*Xnew+Coeff2(3)*Xnew.^2+Coeff2(4)*Xnew.^3+...
    Coeff2(5)*Xnew.^4+Coeff2(6)*Xnew.^5,'-')
%% Validation set
DataNew=[30	166000
26	78000
58	310000
29	100000
40	260000
27	150000
33	140000
61	220000
27	86000
48	276000];
Xn=DataNew(:,1);
Yn=DataNew(:,2);
figure
plot(X,Y,'d')
hold on; plot(Xn,Yn,'s'); legend('Training', 'Validation')
% figure; hold on
% plot(X,Y,'d')
% plot(Xnew,Coeff2(1)+Coeff2(2)*Xnew+Coeff2(3)*Xnew.^2+Coeff2(4)*Xnew.^3+...
%     Coeff2(5)*Xnew.^4+Coeff2(6)*Xnew.^5,'-')
% plot(Xn,Yn,'s');
% figure; hold on
% plot(X,Y,'d')
% plot(Xnew,Coeff1(1)+Coeff1(2)*Xnew,'-')
% plot(Xn,Yn,'s');

[Xn,ord]=sort(Xn);
Yn=Yn(ord);
Mat=[ones(size(Xn)), Xn];
Error1n=norm( Yn-Mat*Coeff1)
Mat=[ones(size(Xn)), Xn, Xn.^2, Xn.^3, Xn.^4, Xn.^5];
Error2n=norm( Yn-Mat*Coeff2)
%% 2nd order Fit
Mat=[ones(size(X)), X, X.^2];
Coeff3=Mat\Y
Error3=norm( Y-Mat*Coeff3)
figure; hold on;
plot(X,Y,'d')
Xnew=linspace(X(1),X(end),100)';
plot(Xnew,Coeff3(1)+Coeff3(2)*Xnew+Coeff3(3)*Xnew.^2,'-')
Mat=[ones(size(Xn)), Xn, Xn.^2];
Error3n=norm( Yn-Mat*Coeff3)
% figure; hold on;
% plot(X,Y,'d')
% plot(Xnew,Coeff3(1)+Coeff3(2)*Xnew+Coeff3(3)*Xnew.^2,'-')
% plot(Xn,Yn,'s');




