function varargout=GetPatch(data,pos)
%% GetPatch
% FORMAT data = GetPatch(data,pos)
% Takes a relatively simple data strucure with filenames
% etc, and converts to a data structure with NIfTI headers etc.
% This allows data to be read in more effectively via:
%
% FORMAT [F,C] = GetPatch(data,pos)
% Reads image data (with possible jitter) into F. C will
% contain a vector of codes.
%
%_______________________________________________________________________
% Copyright (C) 2019-2020 Wellcome Centre for Human Neuroimaging

if nargin==1
    [varargout{1:nargout}] = initialise_data(data);
else
    [varargout{1:nargout}] = get_patch(data,pos);
end


function [data] = initialise_data(data)
data               = fname2nii(data);
[data.dm,data.mat] = get_dimmat(data);
[data.ind, data.p] = get_ind(data);


function data = fname2nii(data)
% Replace filenames with NIfTI headers
for n=1:size(data.dat,1)
    for m=1:numel(data.dat(n).view)
        if ~isempty(data.dat(n).view(m).image)
            data.dat(n).view(m).image = nifti(data.dat(n).view(m).image);
        end
    end
end


function [dm,mat] = get_dimmat(data)
% Check dimensions and orientations
mat = [];
dm  = [];
for n=1:size(data.dat,1)
    for m=1:numel(data.dat(n).view)
        if ~isempty(data.dat(n).view(m).image)
            if isempty(dm)
                dm  = size(data.dat(n).view(m).image(1).dat,[1 2 3]);
                mat = data.dat(n).view(m).image.mat;
            end
            for m1=1:numel(data.dat(n).view(m).image)
                if ~all(size(data.dat(n).view(m).image(m1).dat,[1 2 3])==dm) ||...
                   ~all(data.dat(n).view(m).image(m1).mat(:) == mat(:))
                    error('Incompatible dimensions and/or orientations.');
                end
            end
        end
    end
end


function [ind,pw] = get_ind(data)
M = numel(data.code);
N = 0;
for n=1:numel(data.dat)
    N = N + prod(data.dat(n).jitter*2+1);
end
ind = false(N,M);
pw  = ones(N,1);
for m=1:M
    o   = 0;
    for n=1:numel(data.dat)
        if ~isempty(data.dat(n).view(m).image)
            jit            = data.dat(n).jitter;
            p              = prod(jit*2+1);
            ind(o+(1:p),m) = true;
            sd             = max(data.dat(n).sd,eps);
            [x1,x2,x3]     = ndgrid(-jit(1):jit(1),-jit(2):jit(2),-jit(3):jit(3));
            d2             = x1.^2 + x2.^2 + x3.^2;
            pw(o+(1:p))    = exp((-0.5/sd.^2)*d2(:));
            o              = o+p;
        end
    end
end


function [F,C] = get_patch(data,pos)
Fc = cell(size(data.dat,1),1);
for n=1:size(data.dat,1)
    Fc{n} = get_dat(data.dat(n),data.dm,pos);
end
d2 = zeros(1,32);
d1 = zeros(1,32);
for n=1:numel(Fc)
    for m=1:numel(Fc{n})
        d1(m) = max(d1(m),size(Fc{n}{m},4));
        d2(m) = d2(m)+size(Fc{n}{m},5);
    end
end
d2 = d2(d2>0);
d1 = d1(d2>0);
F  = cell(1,numel(d2));
dm = cellfun(@numel,pos);
for m=1:numel(d2)
    F{m} = zeros([dm d1(m) d2(m)]);
    len  = 0;
    for n=1:numel(Fc)
        lenn = size(Fc{n}{m},5);
        if ~isempty(Fc{n}{m})
            lenn = size(Fc{n}{m},5);
            F{m}(:,:,:,:,len+(1:lenn)) = Fc{n}{m};
        end
        len = len+lenn;
    end
end

C = cell(size(F));
for m=1:numel(C)
    switch data.code(m)
    case 1
        C{m} = [1:size(F{m},4) 0]';
    case 2
        [F{m},C{m}] = OneHot(F{m});
    end
    dm   = [size(F{m}) 1 1];
    F{m} = reshape(F{m},[dm(1)*dm(2)*dm(3),dm(4),dm(5)]);
end


function f = get_dat(dat,dm,pos)
f    = cell(1,numel(dat.view));

% Determine the field of view of what needs to be read from file
jit  = dat.jitter;
pos1 = {(min(pos{1})-jit(1)):(max(pos{1})+jit(1)),...
        (min(pos{2})-jit(2)):(max(pos{2})+jit(2)),...
        (min(pos{3})-jit(3)):(max(pos{3})+jit(3))};
%dm  = mat2cell(dm(1:3),1,[1 1 1]);
dm   = num2cell(dm(1:3));
pos1 = cellfun(@Bound,pos1,dm,'UniformOutput',false);

pos0 = {pos{1}-pos{1}(1)+jit(1)+1, pos{2}-pos{2}(1)+jit(2)+1, pos{3}-pos{3}(1)+jit(3)+1};

view = dat.view;
for m=1:numel(view)
    f{m} = ReadPatchData(view(m),pos1);
    if ~isempty(f{m})
        f{m} = jitter(f{m},pos0,jit);
    else
        f{m} = zeros(zeros(1,5),'single');
    end
end


function f = ReadPatchData(view,pos1)
% Read the data from the image file
dm   = cellfun(@numel,pos1);
f    = zeros([dm numel(view.image)],'single');
for m=1:numel(view.image)
    if ~isempty(view.image(m))
        f(:,:,:,m) = view.image(m).dat(pos1{:});
    end
end


function x = Bound(x,d)
% Deal with boundary conditions when attempting to sample
% points outside the field of view (ie where x<1 or x>d).

%% One form of boundary condition
% x = mod(floor((x-1)./(d-1)),2).*mod(2*d-x-1,2*d-2) + (1-mod(floor((x-1)./(d-1)),2)).*mod(x-1,2*d-2)+1;

%% The boundary condition actually used
x   = mod(floor((x-1)./d),2).*mod(2*d-x,2*d) + (1-mod(floor((x-1)./d),2)).*mod(x-1,2*d)+1;


function f1 = jitter(f,pos0,jit)
dm = [size(f,1) size(f,2) size(f,3) size(f,4)];
n  = jit*2+1;
f1 = zeros([dm(1:3)-2*jit dm(4) prod(n)]);
o  = 0;
for i=-jit(1):jit(1)
    for j=-jit(2):jit(2)
        for k=-jit(3):jit(3)
            pos1 = {pos0{1}+i,pos0{2}+j,pos0{3}+k};
            o    = o + 1;
            for m=1:size(f,4)
                f1(:,:,:,m,o) = f(pos1{:},m);
            end
        end
    end
end


function [F1,u] = OneHot(F)
dm = size(F);
u  = unique(F(:));
F1 = zeros([dm(1:3),numel(u)-1,dm(5)],'single');
for n=1:size(F,5)
    tmp = F(:,:,:,:,n);
    tmp = tmp(:);

    for l=1:(numel(u)-1)
        F1(:,:,:,l,n) = reshape(single(tmp==u(l)),dm(1:3)); %*0.99+0.01/numel(u);
    end
end
F1(~isfinite(F1)) = 0;

