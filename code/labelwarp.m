
% warp label before training

Ptheta = spm_select(Inf,'^y_.*\.nii'); %warp
Plab   = spm_select(size(Ptheta,1),'.*\.img'); %label

Ntheta = nifti(Ptheta);
Nlab   = nifti(Plab);

for i=1:numel(Ntheta)
    f0    = uint8(Nlab(i).dat(:,:,:));
    theta = squeeze(single(Ntheta(i).dat(:,:,:,:,:)));
    theta = AdjustTheta(theta, Nlab(i).mat);
    f1    = WarpLabels(f0,theta);

    [pth,nam,ext] = fileparts(Nlab(i).dat.fname);
    Nii   = Nlab(i);
%     Nii.dat.fname = fullfile('D:\Documents\Yu\Fusion-Challenge\testing-images',['w' nam ext]);
    Nii.dat.fname = fullfile('D:\Documents\Yu\40 LPBA\image',['w' nam ext]);
    Nii.dat.dim   = Ntheta(i).dat.dim(1:3);
    Nii.dat.dtype = 'UINT8';
    Nii.dat.scl_slope = 1;
    Nii.dat.scl_inter = 0;
    Nii.descrip = 'Warped labels';
    Nii.mat     = Ntheta(i).mat;
    create(Nii);
    Nii.dat(:,:,:) = f1;
end