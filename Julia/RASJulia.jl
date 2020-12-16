using LinearAlgebra
using Random


function RASJulia(X::Array{Float64,2},bw::Float64,lambda::Float64,c::Float64,t::Float64,epsilon::Float64,updating::Bool,nbFF::Int64)

N::Int64 = size(X,1);

#  STEP 1: RANDOM FOURIER FEATURES
B = Random_Fourier(X,bw,nbFF);
Id = Diagonal(ones(nbFF,nbFF));
b = cholesky(B'*B+N*lambda*Id); 
F = (b.U)'\(B'); # P = F'*F
B = 0; b = 0; Id = 0; 
    
lev_score::Array{Float64,2} = (sum(F.^2,dims=1));

## STEP 2: Run the sampling algorithm

idS = zeros(Int64,N,1); 
weights = zeros(N,1); 
nmbSamples::Int64 = 0;
epsilon::Float64 = 1e-10; 

PS = Array{Float64,2}(undef,N,nbFF);
R = Array{Float64,2}(undef,1,1);

p_i::Float64 = 0;


for i = 1:N 

    if mod(i,1000) == 1
    	println("Iteration: ",i)
        println("Number of landmarks sampled: ", nmbSamples);
    end
                    
    if nmbSamples == 0
        p_i = min( 1, (1+t)*c*(1/epsilon)*lev_score[i]);

        if rand(1)[1] < p_i
            nmbSamples = 1;
            idS[nmbSamples] = i;
            weights[nmbSamples] = 1/sqrt(p_i);
            R[1,1] = (lev_score[i] .+ epsilon*weights[1].^(-2))^(-1);
            PS[:,1] = F'*F[:,idS[nmbSamples]];
        end 
    else
        
        @views PS_i = PS[i,1:nmbSamples];  
        temp = sum((PS_i'*R).*PS_i',dims=2);

        
        @inbounds p_i = min( 1, (1+t)*c*(1/epsilon)*max(0,lev_score[i] - temp[1]) ); 

        if rand(1)[1] < p_i

            nmbSamples = nmbSamples + 1;
            @inbounds idS[nmbSamples] = i;
            weights[nmbSamples] = 1/sqrt(p_i);


	    @views F_sliced = F[:,idS[nmbSamples]];
            PS[:,nmbSamples] = (F_sliced'*F)';#F'*F_sliced;
                   
             if updating #Update inverse

                 @inbounds newA = PS[i,1:nmbSamples];
                 newA[end,end] = newA[end,end]+ epsilon*weights[nmbSamples].^(-2);
                 R = updateLS(R,newA);
             else
                @inbounds SPS = F[:,idS[1:nmbSamples]]'*F[:,idS[1:nmbSamples]];
                R = (SPS + diagm(0=>vec(epsilon*weights[1:nmbSamples].^(-2))))\diagm(0=>vec(ones(nmbSamples,1)));
             end             
        end      
    end
end


return idS, nmbSamples

end

################# Utilities #################

function  gaussianRandomFeatures(D::Int64, p::Int64,gamma)   
    # sample random Fourier directions and angles
    Omega = sqrt(2*gamma)*randn(p,D); # RVs defining RFF transform
    beta = rand(1,D)*2*pi; 
    return Omega, beta
end


function transformFeatures( X::Array{Float64,2}, Omega::Array{Float64,2}, beta::Array{Float64,2} )

    D = size(Omega,2);
    Z = cos.(X*Omega .+ beta)*sqrt(2/D);
    
    return Z
end


function Random_Fourier(X::Array{Float64,2},bw::Float64,nmbFeatures::Int64)

n, p = size(X);
gamma = 1/(2*bw^2);
Omega, beta  = gaussianRandomFeatures(nmbFeatures,p,gamma);
Z = transformFeatures(X, Omega, beta ); # K = Z*Z';
return Z

end

function updateLS(K_inv::Array{Float64,2},newA::Array{Float64,1})
 
# Init
 a = newA[1:end-1];
 alpha = newA[end];
 
# Update K_inv 
 B1 = K_inv .+ (K_inv*a*a'*K_inv)/(alpha - a'*K_inv*a);
 B2 = -K_inv*a/(alpha - a'*K_inv*a);
 B3 = -a'*K_inv/(alpha - a'*K_inv*a);
 B4 = 1/(alpha-a'*K_inv*a);
 K_inv = [B1 B2;B3 B4];
 
return K_inv
    
end



