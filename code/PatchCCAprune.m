function model = PatchCCAprune(model)
% Prune the model
% FORMAT model = PatchCCAprune(model)
% model - The learned model from PatchCCAtrain
%
% Takes a fitted model, orthogonalise and remove
% irrelevent latent variables.


fprintf('Pruning:    ');
for p3=1:size(model,3)
    for p2=1:size(model,2)
        for p1=1:size(model,1)
            if ~isempty(model(p1,p2,p3).Z)
                model(p1,p2,p3) = Orthogonalise(model(p1,p2,p3));
            end
        end
    end
    fprintf('.');
end
fprintf('\n');




function patch = Orthogonalise(patch)
Z       = patch.Z;
V       = patch.V;
mod     = patch.mod;
p       = patch.p;
%EZZ    = Z*Z'+V;   % Expectation of Z'*Z
EZZ     = Z*bsxfun(@times,p,Z')+V;
[~,~,R] = svd(EZZ); % Rotation to diagonalise EZZ 
Z       = R'*Z;     % Rotate the matrices.
V       = R'*V*R;
for l=1:numel(mod)
    Nvox     = size(mod(l).W,1);
    M        = size(mod(l).W,2);
    K        = size(mod(l).W,3);
    mod(l).W = reshape(reshape(mod(l).W,[Nvox*M,K])*R,[Nvox,M,K]);
end
nz  = sqrt(sum(Z.^2,2)/size(Z,2));
ind = nz>0.001;
Z   = Z(ind,:);
V   = V(ind,ind);
for l=1:numel(mod)
    mod(l).W = mod(l).W(:,:,ind);
end
patch.mod = mod;
patch.Z   = Z;
patch.V   = V;

