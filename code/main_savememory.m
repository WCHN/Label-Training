
addpath('D:\Documents\spm')  

rmpath('D:\Documents\spm\compat')

datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
% datadir = 'D:\Documents\Yu\label_unlabel';

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


tic

% tridx = [1:10 16:565];

% tridx = 1:15;
% data.dat= data.dat(tridx);

sett.nit0    = 4;    
model = PatchCCAtrain(data,sett); % Run the fitting (takes hours)
model = PatchCCAprune(model);
sett.nit0    =6;                 % Number of outer iterations 
model = PatchCCAtrain(data,sett,model); % Run the fitting (takes hours)
model = Patch2NN(model);  

save('D:\Documents\Yu\model2NN_savememory.mat','model','sett','-v7.3'); 

toc
%%
%------------validation---------------

% datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
% validx=11:15;
% 
% files1  = spm_select('FPList',datadir,'^wc1.*\.nii$');               
% F1 = nifti(files1);
% 
% 
% files2  = spm_select('FPList',datadir,'^wc2.*\.nii$');
% F2 = nifti(files2);
% 
% 
% 
% 
% load('D:\Documents\Yu\model2NN_savememory.mat');
% 
% files_y  = spm_select('FPList',datadir,'^y.*\.nii$'); 
% files_w  = spm_select('FPList',datadir,'^w.*glm\.nii$');
% 
% Ntheta = nifti(files_y);
% Ntheta=Ntheta(validx);
% Nlab   = nifti(files_w);
% Nlab=Nlab(validx);
% 
% 
% for i=1:size(Nlab,2)
% F=cat(4,F1(validx(i)).dat,F2(validx(i)).dat);  
% % F=cat(4,F1(i).dat,F2(i).dat);  % concat file array
% [Y,~]=PatchCCAapply(model,F,sett); 
% label_test{i}=Y;
% 
% 
% 
% 
% [pth,nam,ext] = fileparts(Nlab(i).dat.fname);
% Nii   = Nlab(i);
% Nii.dat.fname = fullfile('D:\Documents\Yu\Fusion-Challenge\training-images',['ne' nam ext]);
% % Nii.dat.dim   = Ntheta(i).dat.dim(1:3);
% Nii.dat.dim   = [121 145 53]; % base on label_test size, change base on patch size
% Nii.dat.dtype = 'UINT8';
% Nii.dat.scl_slope = 1;
% Nii.dat.scl_inter = 0;
% Nii.descrip = 'Warped labels';
% Nii.mat     = Ntheta(i).mat;
% create(Nii);
% Nii.dat(:,:,:) = label_test{i}; 
% end
% 
% 
% 
% files_new  = spm_select('FPList',datadir,'^new.*glm\.nii$');
% img=nifti(files_new);
% for n=1:5
% a=dice(img(n).dat(:,:,50),Nlab(n).dat(:,:,50));
% b{n}=nanmean(a(:));
% end



%%

% files_o  = spm_select('FPList',datadir,'^1.*\.nii$');
% files_new  = spm_select('FPList',datadir,'^new.*glm\.nii$');
% b1=files_o(11:15,:); 
% c1=files_y(11:15,:);
% 
% 
% for n = 1:size(Nlab,2)
% 
% matlabbatch{n}.spm.util.defs.comp{1}.inv.comp{1}.def = {c1(n,:)};
% matlabbatch{n}.spm.util.defs.comp{1}.inv.space = {b1(n,:)}; % glm or nii
% matlabbatch{n}.spm.util.defs.out{1}.pull.fnames = {files_new(n,:)};
% 
% matlabbatch{n}.spm.util.defs.out{1}.pull.savedir.saveusr = {'D:\Documents\patch_based_fact'};
% matlabbatch{n}.spm.util.defs.out{1}.pull.interp = -1;
% matlabbatch{n}.spm.util.defs.out{1}.pull.mask = 0;
% matlabbatch{n}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
% matlabbatch{n}.spm.util.defs.out{1}.pull.prefix = 'o';
% 
% end
% 
% spm_jobman('run', matlabbatch);
% 
% 
% 
% da='D:\Documents\patch_based_fact';
% files3=spm_select('FPList',da,'^o.*glm\.nii$');
% Nii=nifti(files3);
% 
% 
% datadir1 = 'D:\Documents\Yu\Fusion-Challenge\training-labels'; % ground truth
% files4 = spm_select('FPList',datadir1,'^1.*\.nii$');
% Nii1=nifti(files4);
% Nii1=Nii1(validx);
% 
% for n=1:size(Nlab,2)
% 
% a=dice(Nii(n).dat(:,:,:),Nii1(n).dat(:,:,:));
% b{n}=nanmean(a(:));
% 
% end

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



load('D:\Documents\Yu\model2NN_savememory.mat');

files_y  = spm_select('FPList',datadir,'^y.*\.nii$'); % this is where the problem is
files_o  = spm_select('FPList',datadir,'^1.*\.nii$');
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
% Nii.dat.dim   = Ntheta(i).dat.dim(1:3);
Nii.dat.dim   = [121 145 63];
Nii.dat.dtype = 'UINT8';
Nii.dat.scl_slope = 1;
Nii.dat.scl_inter = 0;
Nii.descrip = 'Warped labels';
Nii.mat     = Ntheta(i).mat;
create(Nii);
Nii.dat(:,:,:) = label_test{i};

end

files_new  = spm_select('FPList',datadir,'^new.*glm\.nii$');
img=nifti(files_new);

for n=1:20
     i1 = img(n).dat;
     i2 = Nlab(n).dat;
     for i=1:63
     labels = unique(i2(:,:,i));
       if ~isempty(labels)
          i3 = i1(:,:,i);
          i4 = i2(:,:,i);
         for j=1:numel(labels)
           msk1 = i3 == labels(j);
           msk2 = i4 == labels(j);
           dicenum(j,i,n)=dice(msk1,msk2);   % j numebr of class i number of slice n number of subject
         end
       end
      end
end
 
avg=squeeze(mean(dicenum,2));
classavg=mean(avg,2);  
    

%% read generated label map and warp back to original space

% files_new  = spm_select('FPList',datadir,'^new.*glm\.nii$');
% 
% for n = 1:size(F2,2)
% 
% matlabbatch{n}.spm.util.defs.comp{1}.inv.comp{1}.def = {files_y(n,:)};
% matlabbatch{n}.spm.util.defs.comp{1}.inv.space = {files_o(n,:)};
% matlabbatch{n}.spm.util.defs.out{1}.pull.fnames = {files_new(n,:)};
% 
% matlabbatch{n}.spm.util.defs.out{1}.pull.savedir.saveusr = {'D:\Documents\patch_based_fact'};
% matlabbatch{n}.spm.util.defs.out{1}.pull.interp = -1;
% matlabbatch{n}.spm.util.defs.out{1}.pull.mask = 0;
% matlabbatch{n}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
% matlabbatch{n}.spm.util.defs.out{1}.pull.prefix = 'o';
% 
% end
% 
% spm_jobman('run', matlabbatch);
% 
% da='D:\Documents\patch_based_fact';
% files3=spm_select('FPList',da,'^o.*glm\.nii$');
% Nii=nifti(files3);
% 
% datadir1 = 'D:\Documents\Yu\Fusion-Challenge\testing-labels'; % ground truth
% files4 = spm_select('FPList',datadir1,'^1.*\.nii$');
% Nii1=nifti(files4);
% 
% for n=1:size(F2,2)
% 
% a=dice(Nii(n).dat(:,:,:),Nii1(n).dat(:,:,:));
% b{n}=nanmean(a(:));
% 
% end


