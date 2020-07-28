function [W0,W1,W2,V] = NetParams(W,Z,Z2,V,V2,nu0,v0,p)
% FORMAT [W0,W1,W2,V] = NetParams(W,Z,Z2,V,V2,nu0,v0,p)
% W   - Nvox x M x K
% Z   - K  x N
% Z2  - K2 x N
% V   - K  x K
% V2  - K2 x K2
% nu0 - 1  x 1
% v0  - 1  x 1
% p   - N x 1

Nvox = size(W,1);
M    = size(W,2);
K    = size(Z ,1);
K2   = size(Z2,1);
%Ns  = size(Z ,2);
Ns   = sum(p);


%EZZ = [Z*Z'+V, Z*Z2'; Z2*Z', Z2*Z2'+V2];  % Expectation of Z*Z'
EZZ  = [Z *bsxfun(@times,p,Z')+V, Z*bsxfun(@times,p,Z2')
	    Z2*bsxfun(@times,p,Z'),  Z2*bsxfun(@times,p,Z2')+V2];

% Wishart posterior
% P~W(inv(EZZ + (nu0*v0)*eye(K+K2)),(Ns+nu0));
P   = inv(EZZ + (nu0*v0)*eye(K+K2)); % See https://en.wikipedia.org/wiki/Wishart_distribution
nu  = Ns+nu0;
P11 = P(1:K,    1:K  );              % z~N(inv(P11)*(-P12*z2), inv(P11)) 
P12 = P(1:K, K+(1:K2));

% "Bohning bound": Hessian matrix replaced by a global lower bound in the Loewner ordering.
% * Böhning D. Multinomial logistic regression algorithm. Annals of the
%   institute of Statistical Mathematics. 1992 Mar 1;44(1):197-200.
% * Murphy K. Machine learning: a probabilistic approach. Massachusetts
%   Institute of Technology. 2012:1-21.
A = 0.5*(eye(M)-1/(M+1));

% Compute Bohning's approximation to the Hessian
H = P11*(nu-K2);
for i=1:Nvox
    Wi = reshape(W(i,:,:),[M,K]);
    H  = H + Wi'*A*Wi;
end

% Gauss-Newton optimisation can be expressed as a type of Res-Net
% These are the weights required. See NetApply.m for how the weights
% are applied.
W0     = -H\P12*(nu-K2);
W1     =  H\reshape(W,[Nvox*M,K])';
W2     =  W1*kron(A,eye(Nvox))*reshape(W,[Nvox*M,K]);
V      = inv(H);

