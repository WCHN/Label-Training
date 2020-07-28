function Z = PatchCCAlatent(model,F,ind,sett,Z)
% Obtain latent variables for a new image
% Z = PatchCCAlatent(model,F,ind,sett,Z)
% model - The model (computed by PatchCCAtrain and Patch2NN)
% F     - Data to fit the latent variables to
% ind   - Index of the data channel that F corresponds with
% sett  - Settings (uses PatchCCAsettings if missing)
% Z     - Optional starting estimates for latent variables
%

if nargin<3, ind = 1; end
if nargin<4
    sett = PatchCCAsettings;
else
    sett = PatchCCAsettings(sett);
end

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
for it=1:(2*sett.nit0) % Black & White chessboard updates
    pZ=Z;
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
                        Z{p1,p2,p3} = NetApply(Fp,patch.mod(ind).mu,patch.mod(ind).W,patch.W0,patch.W1,patch.W2,z2,Z{p1,p2,p3});
                    end
                end
            end
        end
    end
    ssd=0;
    ss1=0;
    for i=1:numel(Z), ssd = ssd+sum(sum(((Z{i}-pZ{i}).^2))); ss1 = ss1 + sum(sum(Z{i}.^2)); end
    fprintf('%g %g\n', ss1, ssd);
end


function z = GetZ(p1,p2,p3,Z)
if p1>=1 && p1<=size(Z,1) && ...
   p2>=1 && p2<=size(Z,2) && ...
   p3>=1 && p3<=size(Z,3) && ...
   ~isempty(Z{p1,p2,p3})
    z = Z{p1,p2,p3};
else
    z = zeros(0,1);
end

