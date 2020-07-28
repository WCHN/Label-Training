function Ynative = PatchCCAnative(model,Z,phi)
% Reconstruct native-space data
% FORMAT Y = PatchCCAnative(model,Z,phi)
% model - The learned model
% Z     - Cell array of latent variables
% phi   - Mapping from native to template space
% 
% Y     - Native-space patch image
%
%_______________________________________________________________________
% Copyright (C) 2019-2020 Wellcome Centre for Human Neuroimaging

ind = 2; % Use the 2nd view from the model data
dm_temp = cellfun(@max,model(end,end,end).pos); % Size of template-space data

% Find what label values are used
labels = [];
for p=1:numel(model)
    labels = unique([labels(:); model(p).c{ind}(:)]);
end
if max(labels)>255, error('Maximum label value too big.'); end

fprintf('%dx%dx%d (%d labels): ',dm_temp,numel(labels));

% Compute a template-space image (L) of log-sum-exp (lse) results
L = zeros(dm_temp,'single'); % Map of lse values
for p=1:numel(model)         % Loop over patches
    patch     = model(p);                          % Model for current patch
    dm_patch  = cellfun(@numel,patch.pos);         % Dimensions of patch
    lse_patch = GetLSE(Z{p},patch.mod(ind),dm_patch); % log-sum-exp from variables
    L(patch.pos{:}) = lse_patch;                   % Assign to big lse map
end


% Native-space data
dm_native = [size(phi,1) size(phi,2) size(phi,3)]; % Native-space dimensions
Ynative = zeros(dm_native,'uint8');  % Native-space labels
Pnative = zeros(dm_native,'single'); % Native-space maximum label probability


for label=labels' % Loop over labels

    % Generate template-space probability map for this label
    Pl = zeros(dm_temp,'single');
    for p=1:numel(model)                 % Loop over patches
        patch     = model(p);            % Model for current patch
        selection = patch.c{ind}==label; % Where current label is encoded in the patch
        if any(selection)
            m        = find(selection);
            dm_patch = cellfun(@numel,patch.pos);              % Dimensions of patch
            psi      = GetPsi(Z{p},patch.mod(ind),dm_patch,m); % Linear combination of bases 
            Pl(patch.pos{:}) = exp(psi-L(patch.pos{:}));       % Softmax values for this label
        end
    end

    % Warp label probabilities to native space
    Pl = spm_diffeo('pull',Pl,phi);

    % Update native space data
    replace          = find(Pl > Pnative); % Find native voxels to replace
    Pnative(replace) = Pl(replace);        % Update maximum probabilities
    Ynative(replace) = label;              % Update native space labels
    fprintf('.');
end
fprintf('\n');


function psi = GetPsi(z,mod,dm,m)
% Reconstruct psi using only mod.W(:,m,:) and mod.mu(:,m) and reshape to dm
K   = size(mod.W,3);
M   = size(mod.W,2);
if m==M+1
    psi = zeros(dm);
else
    psi = reshape(reshape(mod.W(:,m,:),[prod(dm),K])*z,dm)+reshape(mod.mu(:,m),dm);
end


function lse = GetLSE(z,mod,dm)
% Compute log-sum-exp
K   = size(mod.W,3);
M   = size(mod.W,2);
if M>0
    psi = reshape(reshape(mod.W,[prod(dm)*M,K])*z,[dm M])+reshape(mod.mu,[dm M]);
    mx  = max(max(psi,[],4),0);
    lse = log(sum(exp(bsxfun(@minus,psi,mx)),4)+exp(-mx))+mx;
else
    lse = zeros(dm);
end

