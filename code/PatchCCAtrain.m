function model=PatchCCAtrain(data,sett,model)
% Fit the patch-wise CCA-like model.
% FORMAT model=PatchCCAtrain(data,sett,model)
% data  - a data structure encoding the images used, as well as the
%         amount of jitter etc.
% sett  - a data structure encoding settings.  Fields used are:
%         K       - Number of components to use in the model             [25]
%         nit     - Number of inner iterations for updating mu, W & Z    [5]
%         b0      - Regularisation on W                                  [1.0]
%         nu0     - Wishart degrees of freedom: A ~ W(I v_0 \nu_0, nu_0) [25*7]
%         v0      - Wishart scale parameter:    A ~ W(I v_0 \nu_0, nu_0) [1.0]
%         d1      - Patch-size (currently same in all directions)        [5]
%         nit0    - Outer iterations                                     [20]
%         matname - filename for saving model                            ['']
%         workers - Number of workers in parfor                          [0]
% model - the estimated model
%
%_______________________________________________________________________
% Copyright (C) 2019-2020 Wellcome Centre for Human Neuroimaging

% Default settings
if nargin>=2
    sett = PatchCCAsettings(sett);
else
    sett = PatchCCAsettings;
end

data = GetPatch(data);
dm   = data.dm;
ind  = data.ind;
p    = data.p;

if nargin<3
    % Set up the offsets defining the patches
    d1      = sett.d1;
    %offsets = {1:d1:dm(1), 1:d1:dm(2),50}; % Single slice
    offsets = {1:d1:dm(1), 1:d1:dm(2), 1:d1:dm(3)};

    % Set up a data structure to hold the results
    c       = cell(cellfun(@numel,offsets));
    model   = struct('pos',c,'c',c,'mod',c,'Z',c,'V',c,'p',p);

    for p3=1:numel(offsets{3})
        k   = offsets{3}(p3);
        pos = {[],[],k:min(k+d1-1,dm(3))};
        for p2=1:numel(offsets{2})
            j  = offsets{2}(p2);
            pos{2} = j:min(j+d1-1,dm(2));
            for p1=1:numel(offsets{1})
                i      = offsets{1}(p1);
                pos{1} = i:min(i+d1-1,dm(1));
                model(p1,p2,p3).pos = pos;
            end
        end
    end
end


for it=1:(2*sett.nit0)
    fprintf('%3d-%d:', floor((it+1)/2), rem(it-1,2)+1);
    for p3=1:size(model,3)
   %for p3=15
        fprintf(' %d', p3);
        for p2=1:size(model,2)
            % Collect things together for running a parfor. Can't run
            % parfor over everything because sharing large data structures
            % across nodes is expensive.
            Fs      = cell(1,size(model,1),1);
            patches = cell(1,size(model,1),1);
            Z2s     = cell(1,size(model,1),1);
            V2s     = cell(1,size(model,1),1);
            for p1=1:size(model,1)
                if rem(p1,2)==rem(p2,2)==rem(p3,2)==rem(it,2)
                    patches{p1}       = model(p1,p2,p3);
                    [Fs{p1},C]        = GetPatch(data, patches{p1}.pos);
                    [Z2s{p1},V2s{p1}] = get_neighbours_latent(model,p1,p2,p3);
                    if isempty(patches{p1}.mod)
                        patches{p1}.c = C;
                    end
                end
            end

            if sett.workers==0
                % Use a for loop instead of parfor
                for p1=1:numel(patches)
                    if rem(p1,2)==rem(p2,2)==rem(p3,2)==rem(it,2)
                        patch = patches{p1};
                        F     = Fs{p1};
                        Z2    = Z2s{p1};
                        V2    = V2s{p1};
                        if isempty(patch.mod)
                            [patch.mod,patch.Z,patch.V] = CCAbohning(F,ind,sett,p);
                        else
                            patch   = update_node(patch,F,ind,Z2,V2,sett,p);
                        end
                        patches{p1} = patch;
                    end
                end
            else
                % Run the parfor on the collections of stuff
                parfor(p1=1:numel(patches), sett.workers)
                    if rem(p1,2)==rem(p2,2)==rem(p3,2)==rem(it,2)
                        patch = patches{p1};
                        F     = Fs{p1};
                        Z2    = Z2s{p1};
                        V2    = V2s{p1};
                        if isempty(patch.mod)
                            [patch.mod,patch.Z,patch.V] = CCAbohning(F,ind,sett,p);
                        else
                            patch   = update_node(patch,F,ind,Z2,V2,sett,p);
                        end
                        patches{p1} = patch;
                    end
                end
            end

            % Save the parfor results back to the model
            for p1=1:size(model,1)
                if rem(p1,2)==rem(p2,2)==rem(p3,2)==rem(it,2)
                    model(p1,p2,p3) = patches{p1};
                end
            end
            fprintf('.');
        end
    end
    if isfield(sett,'matname') && it>1
        save(sett.matname, 'model','sett','-v7.3');
    end
    fprintf('\n');
end



function [Z2,V2] = get_neighbours_latent(model,p1,p2,p3)
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



function patch = update_node(patch,F,ind,Z2,V2,sett,p)
% Current patch
Z     = patch.Z;
V     = patch.V;
mod   = patch.mod;
if isempty(Z2), Z2 = zeros(0,size(ind,1)); end

% Expectation of Z*Z' over the central patch and the 6 neighbouring
% patches
EZZ = [Z *bsxfun(@times,p,Z')+V, Z *bsxfun(@times,p,Z2')
       Z2*bsxfun(@times,p,Z'),   Z2*bsxfun(@times,p,Z2')+V2];

% Various dimensions
K   = size(Z,1);
K2  = size(Z2,1);
Ns  = sum(p);

% Expectation of precision matrix, drawn from a Wishart distribution
P   = inv(EZZ + (sett.nu0*sett.v0)*eye(K+K2))*(Ns+sett.nu0);

% Determine distribution of the central patch Z ~ N(Z0,inv(P11))
% This is used as a prior when updating the CCA-like fitting
P11 = P(1:K,  1:K    );
P12 = P(1:K, (1:K2)+K);
Z0  = -P11\P12*Z2;

% Fit the latent variable model again, and update the data structure
[patch.mod,patch.Z,patch.V] = CCAbohning(F,ind,sett,p,mod,Z,Z0,P11);



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

