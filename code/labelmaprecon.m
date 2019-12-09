
% generate label map with trained model

datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';

files1    = spm_select('FPList',datadir,'^wc1.*\.nii$');
F1 = nifti(files1);
files2    = spm_select('FPList',datadir,'^wc2.*\.nii$');
F2 = nifti(files2);


tic

for i=1:15
F=cat(4,F1(i).dat,F2(i).dat);  
[Y,~]=PatchCCAapply(model,F,sett); % Y is label map and P is prob
label_tr{i}=Y;
end

files3= spm_select('FPList',datadir,'^w.*glm\.nii$');
Nii=nifti(files3);
Nii_1=Nii(1:15);


for j=1:15
    Z=label_tr{j};
    score_tr{j}=dice(double(Z(:,:,50)'), Nii_1(j).dat(:,:,50)');
end

tr_mean=mean([score_tr{:}]);
    

toc
