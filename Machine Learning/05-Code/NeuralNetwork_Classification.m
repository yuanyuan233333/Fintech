clear all;
%close all;
% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by NPRTOOL
%
% This script assumes these variables are defined:
%
%[inputs,targets] = cancer_dataset;
rng(10)
inputs= floor(1000*rand(1,10^6));
targets=[(inputs>500); (inputs<=500)];
    
% Create a Pattern Recognition Network
hiddenLayerSize = [3 2];
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
[net,tr] = train(net,inputs,targets);
net.LW

% View the Network
view(net)

% Test the Network
inputs=floor(1000*rand(1,10^6));
targets=[(inputs>500); (inputs<=500)];
outputs=net(inputs);
figure, plotconfusion(targets,outputs)


