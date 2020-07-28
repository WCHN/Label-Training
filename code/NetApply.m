function z = NetApply(F,mu,W,W0,W1,W2,z2,z,nit)
% Update latent variables for a patch
% FORMAT z = NetApply(F,mu,W,W0,W1,W2,z2,z,nit)
%     F   - categorical image data           (Nvox x M)
%     W   - basis functions for decoding     (Nvox x M x K)
%     mu  - mean basis function for decoding (Nvox x M)
%     W0  - precomputed encoding matrix      (K    x K2)
%     W1  - precomputed encoding matrix      (K    x Nvox*M)
%     W2  - precomputed encoding matrix      (K    x K)
%     z2  - expectations of neighbouring latent variables (K2 x 1)
%     z   - expectations of latent variables              (K  x 1)
%     nit - Number of iterations to do [5]
%
% See Patch2NN.m for how W0, W1 and W2 are created.
%_______________________________________________________________________
% Copyright (C) 2019-2020 Wellcome Centre for Human Neuroimaging

if nargin<9, nit = 5; end

Nvox  = size(W,1);
M     = size(W,2);
K     = size(W,3);
if K==0, return; end

% Variational Bayes optimisation expressed as a type of Res-Net
msk    = ~isfinite(F);
F(msk) = 0;
t      = W1*F(:) + W0*z2; % First fully-connected layer
for layer=1:nit
    oz     = z;
    Psi    = reshape(reshape(W,[Nvox*M,K])*z,[Nvox M])+mu; % Psi = W*z+mu fully-connected layer
    P      = SoftMax(Psi,2);                               % SoftMax layer
    P(msk) = 0;
    z      = t + W2*z - W1*P(:);                           % Addition/concatenation layer
    if norm(z-oz)/norm(z)<1e-6, break; end
end

function P = SoftMax(Psi,d)
mx = max(Psi,[],d);
E  = exp(bsxfun(@minus, Psi, mx));
P  = bsxfun(@rdivide, E, sum(E,d)+exp(-mx));


