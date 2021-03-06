function Z = PatchCCAlatent(model,F,ind,sett,Z)
% Obtain latent variables for a new image
% Z = PatchCCAlatent(model,F,ind,sett,Z)
% model - The model (computed by PatchCCAtrain and Patch2NN)
% F     - Data to fit the latent variables to (3D/4D)
% ind   - Index of the data channel that F corresponds with
% sett  - Settings (uses PatchCCAsettings if missing)
% Z     - Optional starting estimates for latent variables
%
%_______________________________________________________________________
% Copyright (C) 2019-2020 Wellcome Centre for Human Neuroimaging

if nargin<3, ind = 1; end
if nargin<4
    sett = PatchCCAsettings;
else
    sett = PatchCCAsettings(sett);
end
nit0 = sett.nit0;
nit  = sett.nit;

if ~isfield(model,'W0'), error('First need to run Patch2NN.'); end

% Needs some error checking for dimensions etc
if nargin<5
    % Initial allocation of cell array of latent variables
    Z     = cell(size(model));
    for p=1:numel(model)
        if ~isempty(model(p).mod)
            Z{p} = zeros([size(model(p).mod(1).W,3),1],'single');
        end
    end
end

% Estimate most probable latent variables
for it=1:(2*nit0) % Black & White chessboard updates
    if sett.verb && rem(it,2)==1, dZ = 0; end;
    for p3=1:size(model,3) % Loop over z
        for p2=1:size(model,2) % Loop over y
            for p1=1:size(model,1) % Loop over x
                if rem(p1,2)==rem(p2,2)==rem(p3,2)==rem(it,2) % Is it a black/white patch
                    patch = model(p1,p2,p3);             % Get the patch
                    if ~isempty(patch.mod)               % Has it been modelled
                        dm  = cellfun(@numel,patch.pos); % Patch dimensions
                        Fp  = F(patch.pos{:},:);         % Get the image patch
                        Fp  = reshape(Fp,[prod(dm) size(F,4)]); % Convert 3D patches to column vectors

                        z2  = [GetZ(p1  ,p2  ,p3+1, Z)
                               GetZ(p1  ,p2  ,p3-1, Z)
                               GetZ(p1  ,p2+1,p3  , Z)
                               GetZ(p1  ,p2-1,p3  , Z)
                               GetZ(p1+1,p2  ,p3  , Z)
                               GetZ(p1-1,p2  ,p3  , Z)]; % Neighbouring latent variables (of other colour)
                        % Estimate latent variables, conditional on neighbouring latent variables
                        if sett.verb, oZ = Z{p1,p2,p3}; end
                        Z{p1,p2,p3} = NetApply(Fp,patch.mod(ind).mu,patch.mod(ind).W,...
                                               patch.W0,patch.W1,patch.W2,z2,Z{p1,p2,p3},nit);
                        if sett.verb, dZ = dZ + sum((oZ(:)-Z{p1,p2,p3}(:)).^2); end
                    end
                end
            end
        end
    end
    if sett.verb && ~rem(it,2), fprintf(' %-.2e', dZ); end
end
if sett.verb, fprintf('\n'); end


function z = GetZ(p1,p2,p3,Z)
if p1>=1 && p1<=size(Z,1) && ...
   p2>=1 && p2<=size(Z,2) && ...
   p3>=1 && p3<=size(Z,3) && ...
   ~isempty(Z{p1,p2,p3})
    z = Z{p1,p2,p3};
else
    z = zeros(0,1);
end

