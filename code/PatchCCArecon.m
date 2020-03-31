function [Y,P]=PatchCCArecon(model,Z,ind)
% Reconstruct data
% FORMAT [Y,P]=PatchCCArecon(model,Z,ind)
% model - The learned model
% Z     - Cell array of latent variables
% ind   - Index of the data-type to reconstruct

if nargin<3, ind = 1; end
msk = cellfun(@(c) ~isempty(c), {model.c});
nc  = max(cellfun(@(c) max(c{ind}), {model(msk).c}));
dm  = cellfun(@max,model(end,end,end).pos);
Y   = zeros(dm,'uint8'); % Label image of same dimensions
if nargout>=2
    if nc*prod(dm)>2^28, error('Too many labels to generate a probabilistic output.'); end
    P = zeros([dm nc],'single');
end

for p=1:numel(model)      % Loop over patches
    patch = model(p);     % Model for current patch
    if isempty(patch.mod) % No model fitted because labels were all identical
       if ~isempty(patch.c) && ~isempty(patch.c{ind})
           Y(patch.pos{:}) = patch.c{ind}(1); % Assign constant label
       else
           Y(patch.pos{:}) = 0;         % Don't know what to do, so assume 0
       end
    else
       dm  = cellfun(@numel,patch.pos); % Dimensions of patch
       z   = Z{p};                      % Latent variables for this patch
       pp  = GetP(z,patch.mod(ind),dm); % Probabilities from latent variables
       [~,mp] = max(pp,[],4);           % Most probable value
       Y(patch.pos{:}) = patch.c{ind}(mp); % Use lookup table to assign voxels to most probable label
       if nargout>=2
           ind1 = find(patch.c{ind}~=0);
           P(patch.pos{:},patch.c{ind}(ind1)) = pp(:,:,:,ind1);
       end
    end
end


function P = GetP_montecarlo(z,V,mod,dm)
K   = size(mod.W,3);
M   = size(mod.W,2);
Ns  = 1000;
z   = z + sqrtm(V)*randn(size(z,1),Ns); % Note that V needs rescaling
psi = bsxfun(@plus, reshape(reshape(mod.W,[prod(dm)*M,K])*z,[dm M Ns]),...
	            reshape(mod.mu,[dm M]));
P   = mean(SoftMax(psi,4),5);


function P = GetP(z,mod,dm)
K   = size(mod.W,3);
M   = size(mod.W,2);
psi = bsxfun(@plus, reshape(reshape(mod.W,[prod(dm)*M,K])*z,[dm M]),...
	            reshape(mod.mu,[dm M]));
P   = SoftMax(psi,4);


function P = SoftMax(psi,d)
dm    = size(psi);
dm(d) = 1;
psi   = cat(d,psi,zeros(dm));
mx    = max(psi,[],d);
E     = exp(bsxfun(@minus, psi, mx));
P     = bsxfun(@rdivide, E, sum(E,d));

