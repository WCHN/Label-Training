



clear all


datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
files_y  = spm_select('FPList',datadir,'^y.*\.nii$');
files_c1  = spm_select('FPList',datadir,'^c1.*\.nii$');
files_c2  = spm_select('FPList',datadir,'^c2.*\.nii$');
% files_c= cat(1,files_c1,files_c2);
files_w  = spm_select('FPList',datadir,'^w.*glm\.nii$');


for n = 1:15
    
    
    
matlabbatch{1}.spm.util.defs.comp{1}.inv.comp{1}.def = {files_y(n,:)};
matlabbatch{1}.spm.util.defs.comp{1}.inv.space = {files_c1(n,:)};
matlabbatch{1}.spm.util.defs.out{1}.pull.fnames = {files_w(n,:)};
matlabbatch{1}.spm.util.defs.out{1}.pull.savedir.savepwd = 1;
matlabbatch{1}.spm.util.defs.out{1}.pull.interp = 4;
matlabbatch{1}.spm.util.defs.out{1}.pull.mask = 1;
matlabbatch{1}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
matlabbatch{1}.spm.util.defs.out{1}.pull.prefix = 'o1';
 
spm('defaults', 'PET');
spm_jobman('run', matlabbatch);

end

for n = 1:15
    
    
    
matlabbatch{1}.spm.util.defs.comp{1}.inv.comp{1}.def = {files_y(n,:)};
matlabbatch{1}.spm.util.defs.comp{1}.inv.space = {files_c2(n,:)};
matlabbatch{1}.spm.util.defs.out{1}.pull.fnames = {files_w(n,:)};
matlabbatch{1}.spm.util.defs.out{1}.pull.savedir.savepwd = 1;
matlabbatch{1}.spm.util.defs.out{1}.pull.interp = 4;
matlabbatch{1}.spm.util.defs.out{1}.pull.mask = 1;
matlabbatch{1}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
matlabbatch{1}.spm.util.defs.out{1}.pull.prefix = 'o2';
 
spm('defaults', 'PET');
spm_jobman('run', matlabbatch);

end
