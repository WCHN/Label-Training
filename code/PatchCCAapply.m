function [Y,P]=PatchCCAapply(model,F)
nit0 = 5;                                % Number of iterations over volume
dm   = [size(F,1), size(F,2) size(F,3)]; % Image dimensions
Y    = zeros(dm);                        % Label image of same dimensions

% Needs some error checking for dimensions etc

% Initial allocation of cell array of latent variables
Z     = cell(size(model));
for p=1:numel(model)
    if ~isempty(model(p).mod)
       Z{p} = zeros([size(model(p).mod(1).W,3),1],'single');
    end
end

% Estimate most probable latent variables
for it=1:(2*nit0) % Black & White chessboard updates
    for p3=1:size(model,3) % Loop over z
        for p2=1:size(model,2) % Loop over y
            for p1=1:size(model,1) % Loop over x
                if rem(p1,2)==rem(p2,2)==rem(p3,2)==rem(it,2) % Is it a black/white patch
                    patch = model(p1,p2,p3);             % Get the patch
                    if ~isempty(patch.mod)               % Has it been modelled
                        dm  = cellfun(@numel,patch.pos); % Patch dimensions
                        Fp  = F(patch.pos{:},:);         % Get the image patch
                        Fp  = reshape(Fp,[prod(dm) size(F,4)]); % Convert to 3D images to column vectors

                        z2  = [GetZ(p1  ,p2  ,p3+1, Z)
                               GetZ(p1  ,p2  ,p3-1, Z)
                               GetZ(p1  ,p2+1,p3  , Z)
                               GetZ(p1  ,p2-1,p3  , Z)
                               GetZ(p1+1,p2  ,p3  , Z)
                               GetZ(p1-1,p2  ,p3  , Z)]; % Neighbouring latent variables (of other colour)
                        % Estimate latent variables, conditional on neighbouring latent variables
                        Z{p1,p2,p3} = NetApply(Fp,patch.mod(1).mu,patch.mod(1).W,patch.W0,patch.W1,patch.W2,z2,Z{p1,p2,p3});
                    end
                end
            end
        end
    end
end

for p=1:numel(model)      % Loop over patches
    patch = model(p);     % Model for current patch
    if isempty(patch.mod) % No model fitted because labels were all identical
       if ~isempty(patch.c)
           Y(patch.pos{:}) = patch.c(1); % Assign constant label
       else
           Y(patch.pos{:}) = 0;          % Don't know what to do, so assume 0
       end
    else
       dm  = cellfun(@numel,patch.pos); % Dimensions of patch
       z   = Z{p};                      % Latent variables for this patch
       P   = GetP(z,patch.mod(2),dm);   % Probabilities from latent variables
       [~,mp] = max(P,[],4);            % Most probable value
       Y(patch.pos{:}) = patch.c(mp);   % Use lookup table to assign voxels to most probable label
    end
end

% Temporary thing
if nargout>=2
P = zeros(size(Y));
%lkp = [1 2 0];
for p=1:numel(model)
    patch = model(p);
    if isempty(patch.mod)
       P(patch.pos{:}) = 0;
    else
       dm  = cellfun(@numel,patch.pos);
       z   = Z{p};
       T   = GetP(z,patch.mod(1),dm);
       P(patch.pos{:}) = T(:,:,:,1);
    end
end
end

function z = GetZ(p1,p2,p3,Z)
if p1>=1 && p1<=size(Z,1) && ...
   p2>=1 && p2<=size(Z,2) && ...
   p3>=1 && p3<=size(Z,3) && ...
   ~isempty(Z{p1,p2,p3})
    z = Z{p1,p2,p3};
else
    z = [];
end

function P = GetP(z,mod,dm)
K   = size(mod.W,3);
M   = size(mod.W,2);
psi = reshape(reshape(mod.W,[prod(dm)*M,K])*z,[dm M])+reshape(mod.mu,[dm M]);
P   = SoftMax(psi,4);

function P = SoftMax(psi,d)
dm    = size(psi);
dm(d) = 1;
psi   = cat(d,psi,zeros(dm));
mx    = max(psi,[],d);
E     = exp(psi-mx);
P     = E./sum(E,d);

function P = ElogP(z,V,mu,W)
Nvox = size(W,1);
M    = size(W,2);
K    = size(W,3);
psi  = reshape(reshape(W,[Nvox*M,K])*z,[Nvox,M])+mu;
A    = 0.5*(eye(M) - 1/(M+1));
lse  = log(1+sum(exp(psi),2));
S    = exp(psi - lse);
b    = psi*A - S;
c    =  0.5*sum(psi*A.*psi,2) - sum(S.*psi,2) + lse;
Vt   = reshape(permute(W,[1 3 2]),[Nvox*K,M])'...
      *reshape(permute(reshape(reshape(W,[Nvox*M,K])*V,[Nvox,M,K]),[1 3 2]),[Nvox*K,M]);
Else = 0.5*trace(A*(Vt+psi'*psi)) - sum(sum(b.*psi,2),1) + sum(c,1);
L    = [psi zeros(Nvox,1)] - Else;
P    = exp(L);

