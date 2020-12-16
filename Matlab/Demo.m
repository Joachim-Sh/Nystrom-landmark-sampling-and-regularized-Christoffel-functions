clear;
close all;
rng shuffle

addpath('Utils')  

%% Data
% Load here the data

X = csvread('../Data/ToyData.csv');

%N = 10000;
%X1 = randn(N/2,2);
%X2 = randn(N/2,2);
%X = [X1;X2+[3.5 -5]; [5 5] + 0.2*randn(N/100,2); [-5 5] + 0.2*randn(N/500,2); [-4 -4] + 0.3*randn(N/500,2)];
%N = size(X,1);
%ind = randperm(N);
%X = X(ind,:);
%X = zscore(X); %standardisation
 

%%  parameters
bw = 1;
epsilon = 1e-10;
c = 200*epsilon;
lambda = 10^(-6);
t = 0;
nb_FF = 4000;
updating = true;
%% sampling
tic
idS = RAS_RFF(X,bw,c,t,epsilon,lambda,nb_FF,updating);
toc
%% plot
figure;
scatter(X(:,1),X(:,2),'.')
hold on;
scatter(X(idS,1),X(idS,2),'o','filled')


