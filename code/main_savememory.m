
addpath('D:\Documents\spm')  

rmpath('D:\Documents\spm\compat')

% datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
datadir = 'D:\Documents\Yu\label_unlabel';

files  = spm_select('FPList',datadir,'^wc1.*\.nii$');
v        = struct('view', {'',''});
dat      = struct('view',  struct('image',{'',''}),...
                  'jitter',[0 0 0]);

clear data
data.code = [1 2];
data.dat(1:size(files,1),1) = deal(dat);

for n=1:size(files,1)

    % Add the various GM and WM maps (assumed to be saved as wc1*.nii and wc2*.nii)
    file_c1 = deblank(files(n,:)); % wc1 gray matter
    [pth,nam,ext] = fileparts(file_c1);
    file_c2  = fullfile(pth,[nam(1:2) '2' nam(4:end) ext]); %wc2 white matter
    data.dat(n).view(1).image = cat(1,file_c1,file_c2); % concatenate
    data.dat(n).jitter        = [0 0 0]; 

    % Include labels if present (assumed to be saved as w*_glm.nii)
    file_glm = fullfile(pth,['w' nam(4:end) '_glm' ext]);    % generated using labelwarp
    if exist(file_glm,'file')
        data.dat(n).view(2).image = file_glm;
        data.dat(n).jitter        = [1 1 1];   % jitter affect the size of Ws (patch size affect Ws too)
    end
    
end

sett  = PatchCCAsettings;         % Default settings
sett.matname = 'D:\Documents\Yu\Fusion-Challenge\model2.mat'; % File to save trained model into
sett.workers = 8;                 % Parallelise training  
sett.K       = 25;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   % Number of components
sett.nu0     =2000;              % Regularisation (part of Wishart prior) bigger than 25*27=675
sett.d1      = 3;                 % Patch size


tridx = [1:10 16:565];
data.dat= data.dat(tridx);

sett.nit0    = 4;    
model = PatchCCAtrain(data,sett); % Run the fitting (takes hours)
model = PatchCCAprune(model);
sett.nit0    =6;                 % Number of outer iterations 
model = PatchCCAtrain(data,sett,model); % Run the fitting (takes hours)
model = Patch2NN(model);  

save('D:\Documents\Yu\model2NN_savememory.mat','model','sett','-v7.3'); 


%%
%------------validation---------------

datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
validx=11:15;

files1  = spm_select('FPList',datadir,'^wc1.*\.nii$');               
F1 = nifti(files1);
files2  = spm_select('FPList',datadir,'^wc2.*\.nii$');
F2 = nifti(files2);

files_y  = spm_select('FPList',datadir,'^y.*\.nii$'); 
files_w  = spm_select('FPList',datadir,'^w.*glm\.nii$');

Ntheta = nifti(files_y);
Ntheta=Ntheta(validx);
Nlab   = nifti(files_w);
Nlab=Nlab(validx);


for i=1:size(Nlab,2)
F=cat(4,F1(validx(i)).dat,F2(validx(i)).dat);  
[Y,~]=PatchCCAapply(model,F,sett); 
label_val{i}=Y;

[pth,nam,ext] = fileparts(Nlab(i).dat.fname);
Nii   = Nlab(i);
Nii.dat.fname = fullfile('D:\Documents\Yu\Fusion-Challenge\training-images',['ne' nam ext]);
Nii.dat.dim   = Ntheta(i).dat.dim(1:3);
Nii.dat.dtype = 'UINT8';
Nii.dat.scl_slope = 1;
Nii.dat.scl_inter = 0;
Nii.descrip = 'Warped labels';
Nii.mat     = Ntheta(i).mat;
create(Nii);
Nii.dat(:,:,:) = label_val{i}; 
end

files3  = spm_select('FPList',datadir,'^new.*glm\.nii$');
Nii1=nifti(files3);
files_w  = spm_select('FPList',datadir,'^w.*glm\.nii$');
Nlab   = nifti(files_w);

for i=1:size(Nlab,2)
imgn=Nii1(i).dat(:,:,:);
labeln=unique(imgn);
imgt=Nlab(i).dat(:,:,:);
labelt=unique(imgt);

 for j=1:numel(labelt)
   msk1 =imgn== labelt(j);
   msk2 =imgt== labelt(j);
 end
 dicenum(j,i)=dice(msk1,msk2); 
 end


%%
%-----------------testing-----------------

addpath('D:\Documents\spm')  

rmpath('D:\Documents\spm\compat')

datadir = 'D:\Documents\Yu\Fusion-Challenge\testing-images'; 

files1  = spm_select('FPList',datadir,'^wc1.*\.nii$');
v        = struct('view', {'',''});
dat      = struct('view',  struct('image',{'',''}),...
                  'jitter',[0 0 0]);    
                    
F1 = nifti(files1);

files2    = spm_select('FPList',datadir,'^wc2.*\.nii$');
F2 = nifti(files2);

files_y  = spm_select('FPList',datadir,'^y.*\.nii$'); 
files_w  = spm_select('FPList',datadir,'^w.*glm\.nii$');

Ntheta = nifti(files_y);
Nlab   = nifti(files_w);

for i=1:size(F1,2)
    
F=cat(4,F1(i).dat,F2(i).dat);  
[Y,~]=PatchCCAapply(model,F,sett); 
label_test{i}=Y;

[pth,nam,ext] = fileparts(Nlab(i).dat.fname);
Nii   = Nlab(i);
Nii.dat.fname = fullfile('D:\Documents\Yu\Fusion-Challenge\testing-images',['ne' nam ext]);
Nii.dat.dim   = Ntheta(i).dat.dim(1:3);
Nii.dat.dtype = 'UINT8';
Nii.dat.scl_slope = 1;
Nii.dat.scl_inter = 0;
Nii.descrip = 'Warped labels';
Nii.mat     = Ntheta(i).mat;
create(Nii);
Nii.dat(:,:,:) = label_test{i};

end

datadirl = 'D:\Documents\Yu\Fusion-Challenge\testing-labels'; 
files_o  = spm_select('FPList',datadirl,'^1.*\.nii$');
files_new  = spm_select('FPList',datadir,'^new.*glm\.nii$');

for n = 1:size(F2,2)

matlabbatch{n}.spm.util.defs.comp{1}.inv.comp{1}.def = {files_y(n,:)};
matlabbatch{n}.spm.util.defs.comp{1}.inv.space = {files_o(n,:)};
matlabbatch{n}.spm.util.defs.out{1}.pull.fnames = {files_new(n,:)};

matlabbatch{n}.spm.util.defs.out{1}.pull.savedir.saveusr = {'D:\Documents\patch_based_fact'};
matlabbatch{n}.spm.util.defs.out{1}.pull.interp = -1;
matlabbatch{n}.spm.util.defs.out{1}.pull.mask = 0;
matlabbatch{n}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
matlabbatch{n}.spm.util.defs.out{1}.pull.prefix = 'o';

end

spm_jobman('run', matlabbatch);

dir='D:\Documents\patch_based_fact';
files3=spm_select('FPList',dir,'^o.*glm\.nii$');
Nii1=nifti(files3); 
Nlab   = nifti(files_o);


for i=1:size(Nlab,2)
imgn=Nii1(i).dat(:,:,:);
labeln=unique(imgn);
imgt=Nlab(i).dat(:,:,:);
labelt=unique(imgt);

 for j=1:numel(labelt)
   msk1 =imgn== labelt(j);
   msk2 =imgt== labelt(j);
 end
 dicenum(j,i)=dice(msk1,msk2); 
 end
