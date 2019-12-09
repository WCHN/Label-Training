% Data could be saves as JSON
%spm_jsonwrite('/tmp/data.jsn',data);
%data = spm_jsonread('/tmp/data.jsn');


sett  = PatchCCAsettings;         % Default settings
sett.matname = 'D:\Documents\Yu\Fusion-Challenge\model2.mat'; % File to save trained model into
sett.d1      = 4;                 % Patch size, need to be changed
sett.K       = 25;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   % Number of components
sett.nu0     =1500;              % Regularisation (part of Wishart prior) bigger than 25*27=675
sett.workers = 8;                 % Parallelise training
sett.nit0    = 4;     


% p='D:\Documents\Yu\Fusion-Challenge\training-images\w1036_3_glm.nii';
p='D:\Documents\Yu\Fusion-Challenge\training-images\w1036_3_glm.nii';

Nii=nifti(p);

datadir  = 'D:\Documents\Yu\Fusion-Challenge\training-images'; % Edit accordingly
% datadir  = 'D:\Documents\Yu\label_unlabel'; % Edit accordingly

files1    = spm_select('FPList',datadir,'^wc1.*\.nii$');
F1 = nifti(files1);
files2    = spm_select('FPList',datadir,'^wc2.*\.nii$');
F2 = nifti(files2);


tic


cv = cvpartition(15,'kfold',5);


for fold = 1:cv.NumTestSets
    
% datadir = 'D:\Documents\Yu\label_unlabel'; % Edit accordingly 
datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
files  = spm_select('FPList',datadir,'^wc1.*\.nii$');
v        = struct('view', {'',''});
dat      = struct('view',  struct('image',{'',''}),...
                  'jitter',[0 0 0]);

clear data
data.code = [1 2];
data.dat(1:size(files,1),1) = deal(dat); % copy dat to dat with 15 times

for n=1:size(files,1)

    % Add the various GM and WM maps (assumed to be saved as wc1*.nii and wc2*.nii)
    file_c1 = deblank(files(n,:)); % wc1 gray matter
    [pth,nam,ext] = fileparts(file_c1);
    file_c2  = fullfile(pth,[nam(1:2) '2' nam(4:end) ext]); %wc2 white matter
    data.dat(n).view(1).image = cat(1,file_c1,file_c2); % concatenate
    data.dat(n).jitter        = [0 0 0]; 

    % Include labels if present (assumed to be saved as w*_glm.nii)
    file_glm = fullfile(pth,['w' nam(4:end) '_glm' ext]);
    if exist(file_glm,'file')
        data.dat(n).view(2).image = file_glm;
        data.dat(n).jitter        = [2 2 2];                                                          
%     else 
%         error('no warped label found')
    end
    
end
    
    
    tridx = cv.training(fold);
    teidx = ~tridx;
    
    v1 = find(tridx==1);
    u1 = find(teidx==1);
    
    w=size(u1,1);
    data.dat= data.dat(v1);
    
    
    model = PatchCCAtrain(data,sett); % Run the fitting (takes hours)
    model = PatchCCAprune(model);
    sett.nit0    = 6;                 % Number of outer iterations
    model = PatchCCAtrain(data,sett,model); % Run the fitting (takes hours)
    model = Patch2NN(model);  
   
    save(['D:\Documents\Yu\Fusion-Challenge\model2NN_' num2str(fold) '.mat'],'model','sett','-v7.3');
 
%     save('D:\Documents\Yu\Fusion-Challenge\model2NN.mat','model','sett','-v7.3');

    for i=1:w
        F=cat(4,F1(u1(i)).dat,F2(u1(i)).dat);  
        [Y,~]=PatchCCAapply(model,F,sett); % Y is label map and P is prob
        label{i}=Y;
    end

    score{fold}=ssim(double(Y(:,:,50)'), Nii.dat(:,:,50)');
    
    
end


toc


