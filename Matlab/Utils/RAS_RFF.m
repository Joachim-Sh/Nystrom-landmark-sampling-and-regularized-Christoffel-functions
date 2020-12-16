% Notes: 

function [idS] = RAS_RFF(X,bw,c,t,epsilon,lambda,nb_FF,updating)

%%
rng shuffle;

%% Initialize

N = size(X,1);

%% STEP 1: Approximate Leverage scores using RANDOM FOURIER FEATURES
B = Random_Fourier(X,bw,nb_FF);
b = chol(B'*B+N*lambda*eye(nb_FF));
F = b'\B'; % P = F'*F

%%
%idS = zeros(N,1);
idS = RASv0(F,c,t,epsilon,updating);
nmbSamples = nnz(idS);
idS = idS(1:nmbSamples);


end

%% Utility



