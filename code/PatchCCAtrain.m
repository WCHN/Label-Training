function model=PatchCCAtrain

d1   = 5;  % Patch size
nit0 = 10; % Number of outer iterations
sett = struct('K',32,'nit',5,'b0',1,'nu0',32*7,'v0',16);
%sett = struct('K',20,'nit',5,'b0',0.001,'nu0',1,'v0',10);

%files = spm_select('FPList','data','^w.*_glm\.nii$');
files = spm_select('FPList','data','^wc1.*\.nii$');
C     = cell(size(files,1),2);
for n=1:size(files,1)
    file_c1 = deblank(files(n,:));
    [pth,nam,ext] = fileparts(file_c1);
    file_c2  = fullfile(pth,[nam(1:2) '2' nam(4:end) ext]);
    file_glm = fullfile(pth,['w' nam(4:end) '_glm' ext]);
    C{n,1}   = nifti(char(file_c1,file_c2));
    if exist(file_glm,'file')
        C{n,2} = nifti(file_glm);
    end
end

ind  = ~cellfun(@isempty,C);
ind1 = repmat(ind,27,1);
Nii1 = cat(1,C{:,1});
Nii2 = cat(1,C{:,2});

FA1    = cell(size(Nii1));
FA1(:) = {Nii1(:).dat};
FA2    = cell(size(Nii2));
FA2(:) = {Nii2(:).dat};

dm    = [size(Nii1(1).dat), 1];
dm    = dm(1:3);


offsets = {2:d1:(dm(1)-1), 2:d1:(dm(2)-1),50};
%offsets = {2:d1:(dm(1)-1), 2:d1:(dm(2)-1),2:d1:(dm(3)-1)};

dr      = [numel(offsets{1}),numel(offsets{2}),numel(offsets{3})];
c       = cell(dr);
model   = struct('pos',c,'c',c,'mod',c,'Z',c,'V',c,'W0',c,'W1',c,'W2',c);

for p3=1:numel(offsets{3})
    k = offsets{3}(p3);
    pos = {[],[],k:min(k+d1-1,dm(3)-1)};
    for p2=1:numel(offsets{2})
        j     = offsets{2}(p2);
        pos{2} = j:min(j+d1-1,dm(2)-1);
        for p1=1:numel(offsets{1})
            i      = offsets{1}(p1);
            pos{1} = i:min(i+d1-1,dm(1)-1);
            model(p1,p2,p3).pos = pos;
        end
    end
end

for it=1:(2*nit0)
    for p3=1:size(model,3)
        for p2=1:size(model,2)
            for p1=1:size(model,1)
                if rem(p1,2)==rem(p2,2)==rem(p3,2)==rem(it,2)
                    patch   = model(p1,p2,p3);
                    [F2,c]  = get_label_patch(FA2,patch.pos);
                    F1      = load_patch(FA1,patch.pos,'single');
                    patch.c = c;
                    if numel(c)>1
                        if it<=2
                            [mod,Z,V] = CCAbohning({F1,F2},ind1,sett);
                            patch.mod = mod;
                            patch.Z   = Z;
                            patch.V   = V;
                        else
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

                            if isempty(Z2), Z2 = zeros(0,size(ind1,1)); end
                            Z   = patch.Z;
                            V   = patch.V;
                            mod = patch.mod;
                            K   = size(Z,1);
                            K2  = size(Z2,1);
                            N   = size(Z,2);
                            EZZ = [Z*Z'+V Z*Z2'; Z2*Z' Z2*Z2'+V2];
                            P   = inv(EZZ + (sett.nu0*sett.v0)*eye(K+K2))*(N+sett.nu0);
                            P11 = P(1:K,  1:K    );
                            P12 = P(1:K, (1:K2)+K);
                            Z0  = -P11\P12*Z2;

                            % Fit the latent variable model
                            [mod,Z,V] = CCAbohning({F1,F2},ind1,sett,mod,Z,V,Z0,P11);
                            patch.mod = mod;
                            patch.Z   = Z;
                            patch.V   = V;
                            if it>=(2*nit0)-1
                                [patch.W0,patch.W1,patch.W2] = NetParams(mod(1).W,Z,Z2,V,V2,sett.nu0,sett.v0);
                            end
                        end
                    else
                        patch.mod = [];
                        patch.Z   = zeros(0,size(ind1,1),'single');
                        patch.V   = eye(0);
                    end
                    model(p1,p2,p3) = patch;
                end
            end
            fprintf('.');
        end
        fprintf('\n');
    end
    save model4.mat model
    fprintf('\n');
end

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


function [F,c] = get_label_patch(FA,pos)
X     = load_patch(FA,pos,'uint8');
[F,c] = OneHot(X);

function X = load_patch(FA,pos,cls)
if nargin<3, cls = 'single'; end
dm = [cellfun(@numel,pos), size(FA,2), 27*size(FA,1)];
X  = zeros(dm,cls);
Xc = cell(size(FA,1),size(FA,2));
pos1 = {(min(pos{1})-1):(max(pos{1})+1), (min(pos{2})-1):(max(pos{2})+1), (min(pos{3})-1):(max(pos{3})+1)};
for n=1:size(FA,1)
    for m=1:size(FA,2)
        Xc{n,m} = FA{n,m}(pos1{:});
    end
end
pos0 = {(1:dm(1))+1, (1:dm(2))+1, (1:dm(3))+1};

for i=-1:1
    for j=-1:1
        for k=-1:1
            pos1 = {pos0{1}+i,pos0{2}+j,pos0{3}+k};
            o    = ((k+1)+3*((j+1)+3*(i+1)))*size(FA,1);
            for n=1:size(FA,1)
                for m=1:size(FA,2)
                    X(:,:,:,m,n+o) = Xc{n,m}(pos1{:});
                end
            end
        end
    end
end
X = reshape(X,[prod(dm(1:3)),dm(4:5)]);


function [F,u] = OneHot(X)
dm = size(X);
u  = unique(X(:));
F  = zeros([dm(1),numel(u)-1,dm(3)],'single');
for n=1:size(X,3)
    tmp = X(:,:,n);
    tmp = tmp(:);

    for l=1:(numel(u)-1)
        F(:,l,n) = single(tmp==u(l)); %*0.99+0.01/numel(u);
    end
end
F(~isfinite(F)) = 0;

