%train the model

datadir = 'D:\Documents\Yu\label_unlabel';
% datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
files    = spm_select('FPList',datadir,'^wc1.*\.nii$');
v        = struct('view', {'',''});
dat      = struct('view',  struct('image',{'',''}),...
                  'jitter',[0 0 0],'sd',0);

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
	data.dat(n).sd            = 0.75;
    end
end


sett         = PatchCCAsettings;  % Default settings
sett.matname = fullfile(datadir,'model2.mat'); % File to save trained model into
sett.d1      = 4;                 % Patch size
sett.workers = 8;                 % Parallelise training
sett.K       = 25;                % Number of components
sett.nu0     = 175;               % Regularisation (part of Wishart prior): Number of neighbours (+central)*  Number of components (6+1)*25=175
sett.nit0    = 8;                 % Number of outer iterations
model = PatchCCAtrain(data,sett); % Run the fitting (takes hours)
model = PatchCCAprune(model);     % Prune the model again
model = PatchCCAtrain(data,sett,model); % Run the fitting (takes hours)
model = Patch2NN(model);          % Convert model to NN form
% save(fullfile(datadir,'model2NN.mat','model','sett','-v7.3'));
save('D:\Documents\Yu\model2NN.mat','model','sett','-v7.3'); 


%% testing

datadir = 'D:\Documents\Yu\Fusion-Challenge\testing-images'; 
files1  = spm_select('FPList',datadir,'^wc1.*\.nii$');
F1 = nifti(files1);
files2    = spm_select('FPList',datadir,'^wc2.*\.nii$');
F2 = nifti(files2);
files_y  = spm_select('FPList',datadir,'^y.*\.nii$'); 

datadir1 = 'D:\Documents\Yu\Fusion-Challenge\testing-labels'; 
files_w  = spm_select('FPList',datadir1,'^w.*glm\.nii$');
files_o  = spm_select('FPList',datadir1,'^1.*\.nii$');



Ntheta = nifti(files_y);
Nlab   = nifti(files_w);


for i= 1: numel(Nlab)
    F=cat(4,F1(i).dat,F2(i).dat);  
    [Y,~]=PatchCCAapply(model,F,sett); 
    label{i}=Y;
    [pth,nam,ext] = fileparts(Nlab(i).dat.fname);
    Nii   = Nlab(i); 
    Nii.dat.fname = fullfile('D:\Documents\Yu\Fusion-Challenge\testing-labels',['p_t_' nam ext]);
    Nii.dat.dim   = Ntheta(i).dat.dim(1:3);
    Nii.dat.dtype = 'UINT8'; 
    Nii.dat.scl_slope = 1;
    Nii.dat.scl_inter = 0;
    Nii.descrip = 'predicted test labels';
    Nii.mat     = Ntheta(i).mat;  
    create(Nii);
    Nii.dat(:,:,:) = label{i};
end

%% warp back to native space

files_new  = spm_select('FPList',datadir1,'^p.*glm\.nii$');

for n = 1:numel(Nlab)

matlabbatch{n}.spm.util.defs.comp{1}.inv.comp{1}.def = {files_y(n,:)};
matlabbatch{n}.spm.util.defs.comp{1}.inv.space = {files_o(n,:)};
matlabbatch{n}.spm.util.defs.out{1}.pull.fnames = {files_new(n,:)};
matlabbatch{n}.spm.util.defs.out{1}.pull.savedir.saveusr = {'D:\Documents\Yu\Fusion-Challenge\testing-labels'};
matlabbatch{n}.spm.util.defs.out{1}.pull.interp = -1;
matlabbatch{n}.spm.util.defs.out{1}.pull.mask = 0;
matlabbatch{n}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
matlabbatch{n}.spm.util.defs.out{1}.pull.prefix = 'o';

end

spm_jobman('run', matlabbatch);



%% compute dice

Nlabo   = nifti(files_o);
files_ow  = spm_select('FPList',datadir1,'^o.*glm\.nii$');
Nlabow   = nifti(files_ow);


for i =1:numel(Nlabo)
    
    
img_gt=Nlabo(i).dat(:,:,:);
label_gt=unique(img_gt);

img_new=Nlabow(i).dat(:,:,:);
label_new=unique(img_new);

C = intersect(label_new,label_gt);


 for j=1:numel(C)
   msk1 =img_gt == C(j);
   msk2 =img_new == C(j);
   dicenum(j,i)=dice(msk1,msk2);  

 end


end

dicenum(1,:)=[]; % remove class 0
dice1=mean(dicenum,1)'; % across classes
dice2=mean(dicenum,2);  % across subjects


