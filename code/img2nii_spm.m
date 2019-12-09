%%img2nii.m--------------------------------------------
%Script to convert hdr/img files to nii.
%This script uses SPM function, so you need to install SPM5 or later.
%Kiyotaka Nemoto 05-Nov-2014
 
%select files
f = spm_select(Inf,'img$','Select img files to be converted');
 
%convert img files to nii
for i=1:size(f,1)
  input = deblank(f(i,:));
  [pathstr,fname,ext] = fileparts(input);
    output = strcat(fname,'.nii');
    V=spm_vol(input);
    ima=spm_read_vols(V);
    V.fname=output;
    spm_write_vol(V,ima);
end