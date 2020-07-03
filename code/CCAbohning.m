function [mod,Z,V] = CCAbohning(F,ind,sett,p,varargin)
%% Bohning bound CCA stuff
%% [mod,Z,V] = CCAbohning(F,ind,sett,p)
%%

% F  - Nvox x M x N
% Z  - K x N
% V  - K x K
% mu - Nvox x M
% W  - Nvox x M x K
% A  - K x K
%% 
%% [mod,Z,V] = CCAbohning(F,ind,sett,p,mod,Z,V,Z0,P0)
%
%%
% Figure out the various desired settings.
N     = size(ind,1);

if nargin>=3
    sett = PatchCCAsettings(sett);
else
    sett = PatchCCAsettings;
end

if nargin<=4
    if isa(F,'cell')
        c = cell(1,numel(F));
    else
        c = cell(1);
        F = {F};
    end
    K   = sett.K;
    mod = struct('mu',c,'W',c);
    for l=1:numel(mod)
        if sum(ind(:,l),1)~=size(F{l},3)
            error('Incompatible dimensions (%d ~= %d).', sum(ind(:,l),1), size(F{l},3));
        end
        mod(l).mu = zeros(size(F{l},1), size(F{l},2),'single');
        mod(l).W  = zeros(size(F{l},1), size(F{l},2), K,'single');
    end
    randn('seed',0);
    B0  = eye(K)*sett.b0;
    Z   = randn(K,N,'single');
    Z   = bsxfun(@minus, Z, (Z*p)/sum(p));
else
    mod = varargin{1};
    Z   = varargin{2};
    K   = size(mod(1).W,3);
    B0  = eye(K)*sett.b0;
end

if nargin<7
    V  = eye(K);
else
    V  = varargin{3};
end
if nargin<8
    Z0 = zeros(K,N);
else
    Z0 = varargin{4};
end
if nargin<9
    P0 = eye(K)/sett.v0;
else
    P0 = varargin{5};
end

%%
% Run the iterative variational Bayesian EM algorithm itself.
for iter=1:sett.nit

    % Variational M-step
    for l=1:numel(mod)
        [mod(l).mu,mod(l).W] = UpdateW(F{l}, Z(:,ind(:,l)), V, mod(l).mu, mod(l).W, B0, p(ind(:,l)));
    end

    % Variational E-step
    Hc  = cell(1,numel(mod));
    for l=1:numel(Hc)
        Hc{l} = HessZ(mod(l).W);
    end
    csi = cumsum(ind,1);
    V   = 0;
    for n=1:N
        z  = Z(:,n);
        H  = P0;
        g  = P0*Z0(:,n);
        for l=1:numel(mod)
            if ind(n,l)
                g = g + NumeratorZ(F{l}(:,:,csi(n,l)),z,mod(l).mu,mod(l).W);
                H = H + Hc{l};
            end
        end
        Z(:,n) = H\g;
        V      = V + p(n)*inv(H);
    end
    Z  = bsxfun(@minus, Z, (Z*p)/sum(p));
end

if sett.do_orth
    EZZ     = Z*bsxfun(@times,p,Z') + V;
    [~,~,R] = svd(EZZ); % Rotation to diagonalise EZZ 
    Z       = R'*Z;    % Rotate the matrices.
   %Z0      = R'*Z0;
   %P0      = R'*P0*R;
    V       = R'*V*R;
    for l=1:numel(mod)
        Nvox     = size(F{l},1);
        M        = size(F{l},2);
        mod(l).W = reshape(reshape(mod(l).W,[Nvox*M,K])*R,[Nvox,M,K]);
    end
end


%% UpdateW
% Update the mean ($\bf\mu$) and basis functions ($\bf W$).
%
% See Murphy's textbook.
%%
% * Murphy K. _Machine learning: a probabilistic approach_ . Massachusetts
%   Institute of Technology. 2012:1-21.
function [mu,W] = UpdateW(F,Z,V,mu,W,B,p)
if isempty(mu) return; end
Nvox  = size(F,1);
M     = size(F,2);
N     = size(F,3);
K     = size(W,3);
Ns    = sum(p);
A     = Abohning(M);

%%
% Update $\boldsymbol\mu$.
Vm = inv(Ns*A);                          % Cov mu
for i=1:Nvox
    Fi       = reshape(F(i,:,:),[M,N]);
    msk      = ~isfinite(Fi);
    Psi      = bsxfun(@plus,reshape(W(i,:,:),[M,K])*Z, mu(i,:)');
    R        = bsxfun(@plus,Fi, bsxfun(@minus, A*mu(i,:)', SoftMax(Psi,1)));
    R(msk)   = 0;
    mu(i,:)  = (Vm*(R*p))';              % Update of mu
end


%% 
% Update ${\bf W}$.
Vw  = inv(kron(Z*bsxfun(@times,p,Z')+V,A) + kron(B,eye(M)-1/(M+1)));
for i=1:Nvox
    Fi       = reshape(F(i,:,:),[M,N]);
    msk      = ~isfinite(Fi);
    Psi0     = reshape(W(i,:,:),[M,K])*Z;
    Psi      = bsxfun(@plus, Psi0, mu(i,:)');
    R        = bsxfun(@plus,Fi, bsxfun(@minus, A*Psi0, SoftMax(Psi,1)));
    R(msk)   = 0;
    g        = reshape(R*bsxfun(@times,p,Z'),[M*K,1]); 
    W(i,:,:) = reshape((Vw*g)',[1 M K]); % Update of W
end


%% HessZ
% Compute Bohning's lower bound approximation to the Hessian used for updating
% the approximation to ${\bf z}$.
function H = HessZ(W)
Nvox = size(W,1);
M    = size(W,2);
K    = size(W,3);
A    = Abohning(M);
H    = 0;
for i=1:Nvox
    Wi = reshape(W(i,:,:),[M,K]);
    H  = H + Wi'*A*Wi;
end


%% ComputeWW
% Compute ${\bf W}^T{\bf W}$, accounting for image dimensions etc (unused).
%%
function WW = ComputeWW(W)
Nvox = size(W,1);
M    = size(W,2);
K    = size(W,3);
W    = reshape(W,[Nvox*M, K]);
WW   = W'*W;


%% Abohning
% "Bohning bound": Hessian matrix replaced by a global lower bound in the Loewner ordering.
%
% ${\bf A} = \frac{1}{2}({\bf I}_M - \frac{1}{M+1})$
%%
% * Böhning D. _Multinomial logistic regression algorithm_ . Annals of the
%   institute of Statistical Mathematics. 1992 Mar 1;44(1):197-200.
function A = Abohning(M)
A  = 0.5*(eye(M)-1/(M+1));


%% NumeratorZ
% See Algorithm 21.1 of Murphy's textbook.
%%
% * Murphy K. _Machine learning: a probabilistic approach_ . Massachusetts
%   Institute of Technology. 2012:1-21.
function g = NumeratorZ(Fn,z,mu,W)
if isempty(W), g = zeros([size(W,3),1],'single'); return; end
Nvox = size(Fn,1);
M    = size(Fn,2);
K    = size(W,3);
A    = Abohning(M);
Psi0 = reshape(  reshape(W,[Nvox*M,K])*z,[Nvox,M]);
P    = SoftMax(Psi0+mu,2);
r    = reshape(Fn-P+Psi0*A,[1,Nvox*M]);
g    = reshape(r*reshape(W,[Nvox*M,K]),[K,1]);


%% SoftMax
% Safe softmax over dimension $d$, which prevents over/underflow.
% 
% $$p_k = \frac{\exp \psi_k}{\sum_{c=1}^K \exp \psi_c}$$
%
% With the constraint \psi_K=0
%
% $$p_k = \frac{\exp \psi_k}{1+\sum_{c=1}^{K-1} \exp \psi_c}$$
function P = SoftMax(Psi,d)
mx  = max(Psi,[],d);
E   = exp(bsxfun(@minus,Psi,mx));
P   = bsxfun(@rdivide, E, sum(E,d)+exp(-mx));

%%
%%
