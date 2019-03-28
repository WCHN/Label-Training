function model = Patch2NN(model,ind,sett)
% Convert to neural network form
% FORMAT model = Patch2NN(model,ind,sett)
% model - The learned model from PatchCCAtrain
% ind   - Index of the data channel to optimise from
% sett  - Settings (uses PatchCCAsettings if missing)
%
% Takes a fitted model, and converts to a form that allows
% the distributions of latent variables to be estimated by
% a neural network type formulation.

if nargin<2, ind = 1; end
if ~isfield(model,'Z'), error('Model is already converted to NN form.'); end

if nargin<3
    sett = PatchCCAsettings;
else
    sett = PatchCCAsettings(sett);
end

model(1).W0 = [];
model(1).W1 = [];
model(1).W2 = [];
model(1).Va = [];
model(1).ind = ind;

fprintf('Pruning:    ');
for p3=1:size(model,3)
    for p2=1:size(model,2)
        for p1=1:size(model,1)
            model(p1,p2,p3) = Orthogonalise(model(p1,p2,p3));
        end
    end
    fprintf('.');
end
fprintf('\nConverting: ');

for p3=1:size(model,3)
    for p2=1:size(model,2)
        for p1=1:size(model,1)
            patch   = model(p1,p2,p3);
            Z2  = [GetZ(p1  ,p2  ,p3+1,model)
                   GetZ(p1  ,p2  ,p3-1,model)
                   GetZ(p1  ,p2+1,p3  ,model)
                   GetZ(p1  ,p2-1,p3  ,model)
                   GetZ(p1+1,p2  ,p3  ,model)
                   GetZ(p1-1,p2  ,p3  ,model)];

            V2  = blkdiag(GetV(p1  ,p2  ,p3+1,model),...
                          GetV(p1  ,p2  ,p3-1,model),...
                          GetV(p1  ,p2+1,p3  ,model),...
                          GetV(p1  ,p2-1,p3  ,model),...
                          GetV(p1+1,p2  ,p3  ,model),...
                          GetV(p1-1,p2  ,p3  ,model));

            % Current patch
            Z   = patch.Z;
            V   = patch.V;

            if isempty(Z2), Z2 = zeros(0,size(Z,2)); end

            if ~isempty(patch.mod)
                [patch.W0,patch.W1,patch.W2,patch.Va] = NetParams(patch.mod(ind).W,Z,Z2,V,V2,sett.nu0,sett.v0);
                patch.ind = ind;
            end
            model(p1,p2,p3) = patch;
        end
    end
    fprintf('.');
end
%model = rmfield(model,'Z');
%model = rmfield(model,'V');
fprintf('\n');


function patch = Orthogonalise(patch)
Z       = patch.Z;
V       = patch.V;
mod     = patch.mod;
EZZ     = Z*Z'+V;   % Expectation of Z'*Z
[~,~,R] = svd(EZZ); % Rotation to diagonalise EZZ 
Z       = R'*Z;     % Rotate the matrices.
V       = R'*V*R;
for l=1:numel(mod)
    Nvox     = size(mod(l).W,1);
    M        = size(mod(l).W,2);
    K        = size(mod(l).W,3);
    mod(l).W = reshape(reshape(mod(l).W,[Nvox*M,K])*R,[Nvox,M,K]);
end
nz  = sum(Z.^2,2)/size(Z,2);
ind = nz>sqrt(1/100000);
Z   = Z(ind,:);
V   = V(ind,ind);
for l=1:numel(mod)
    mod(l).W = mod(l).W(:,:,ind);
end
patch.mod = mod;
patch.Z   = Z;
patch.V   = V;


function Z = GetZ(p1,p2,p3,model)
if p1>=1 && p1<=size(model,1) && ...
   p2>=1 && p2<=size(model,2) && ...
   p3>=1 && p3<=size(model,3) && ...
   ~isempty(model(p1,p2,p3).Z)
    Z = model(p1,p2,p3).Z;
else
    Z = [];
end

function V = GetV(p1,p2,p3,model)
if p1>=1 && p1<=size(model,1) && ...
   p2>=1 && p2<=size(model,2) && ...
   p3>=1 && p3<=size(model,3) && ...
   ~isempty(model(p1,p2,p3).V)
    V = model(p1,p2,p3).V;
else
    V = [];
end

