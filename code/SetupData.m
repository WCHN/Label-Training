datadir  = '/home/john/WIP/SGA2/data'; % Edit accordingly
files    = spm_select('FPList',datadir,'^wc1.*\.nii$');
v        = struct('view', {'',''});
dat      = struct('view',  struct('image',{'',''}),...
                  'jitter',[0 0 0]);

clear data
data.code = [1 2];
data.dat(1:size(files,1),1) = deal(dat);

for n=1:size(files,1)

    % Add the various GM and WM maps (assumed to be saved as wc1*.nii and wc2*.nii)
    file_c1 = deblank(files(n,:));
    [pth,nam,ext] = fileparts(file_c1);
    file_c2  = fullfile(pth,[nam(1:2) '2' nam(4:end) ext]);
    data.dat(n).view(1).image = cat(1,file_c1,file_c2);
    data.dat(n).jitter        = [0 0 0];

    % Include labels if present (assumed to be saved as w*_glm.nii)
    file_glm = fullfile(pth,['w' nam(4:end) '_glm' ext]);
    if exist(file_glm,'file')
        data.dat(n).view(2).image = file_glm;
        data.dat(n).jitter        = [1 1 1];
    end
end

% Data could be saves as JSON
%spm_jsonwrite('/tmp/data.jsn',data);
%data = spm_jsonread('/tmp/data.jsn');

sett  = PatchCCAsettings;         % Default settings
sett.matname = 'model.mat';       % File to save trained model into
sett.d1      = 4;                 % Patch size
sett.workers = 10;                % Parallelise training
sett.K       = 16;                % Number of components
sett.nu0     = 1500;              % Regularisation (part of Wishart prior)
sett.nit0    = 8;                 % Number of outer iterations
model = PatchCCAtrain(data,sett); % Run the fitting (takes hours)
model = Patch2NN(model);          % Convert model to NN form

