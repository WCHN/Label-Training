function z = NetApply(F,mu,W,W0,W1,W2,z2,z)

Nvox  = size(W,1);
M     = size(W,2);
K     = size(W,3);
if K==0, return; end

% Gauss-Newton optimisation expressed as a type of Res-Net
msk    = ~isfinite(F);
F(msk) = 0;
t      = W1*F(:) + W0*z2; % First fully-connected layer, F should be wc1 and wc2 at a time (size 27*2)
for layer=1:5
    oz     = z;
    Psi    = reshape(reshape(W,[Nvox*M,K])*z,[Nvox M])+mu; % Psi = W*z+mu fully-connected layer
    P      = SoftMax(Psi,2);                               % SoftMax layer
    P(msk) = 0;
    z      = t + W2*z - W1*P(:);                           % Addition/concatenation layer
    if norm(z-oz)/norm(z)<1e-6, break; end
end

function P = SoftMax(Psi,d)
mx  = max(Psi,[],d);
E   = exp(Psi-mx);
P   = E./(sum(E,d)+exp(-mx));

