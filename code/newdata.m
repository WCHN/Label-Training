% Data could be saves as JSON
%spm_jsonwrite('/tmp/data.jsn',data);
%data = spm_jsonread('/tmp/data.jsn');

datadir = 'D:\Documents\Yu\40 LPBA\image'; % Edit accordingly 
% datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
files  = spm_select('FPList',datadir,'^wc1.*\.nii$');
v        = struct('view', {'',''});
dat      = struct('view',  struct('image',{'',''}),...
                  'jitter',[0 0 0]);
              
files_img= spm_select('FPList',datadir,'^w.*\.img$');

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
    file_c3 = deblank(files_img(n,:)); % wc1 gray matter
    [pth,nam1,ext1] = fileparts(file_c3);
    file_glm = fullfile(pth,['w' nam1(2:end) ext1]);
    if exist(file_glm,'file')
        data.dat(n).view(2).image = file_glm;
        data.dat(n).jitter        = [2 2 2];                                                          

    end
    
end

%% settings

sett  = PatchCCAsettings;         % Default settings
sett.matname = 'D:\Documents\Yu\40 LPBA\model2.mat'; % File to save trained model into
sett.d1      = 4;                 % Patch size, need to be changed
sett.K       = 25;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   % Number of components
sett.nu0     =1500;              % Regularisation (part of Wishart prior) bigger than 25*27=675
sett.workers = 8;                 % Parallelise training
sett.nit0    = 4;     
tic
model = PatchCCAtrain(data,sett); % Run the fitting (takes hours)
model = PatchCCAprune(model);
sett.nit0    = 6;                 % Number of outer iterations
model = PatchCCAtrain(data,sett,model); % Run the fitting (takes hours)
model = Patch2NN(model);          % Convert model to NN form
save('D:\Documents\Yu\40 LPBA\model2NN.mat','model','sett','-v7.3');

toc
%% training
    
% tridx = [1:10 16:565];
% 
% data.dat= data.dat(tridx);
% 
% model = PatchCCAtrain(data,sett); % Run the fitting (takes hours)
% model = PatchCCAprune(model);
% sett.nit0    = 6;                 % Number of outer iterations
% model = PatchCCAtrain(data,sett,model); % Run the fitting (takes hours)
% model = Patch2NN(model);  
% 
% save('D:\Documents\Yu\Fusion-Challenge\model2NN.mat','model','sett','-v7.3');
% 
% %% validation
% 
% files1    = spm_select('FPList',datadir,'^wc1.*\.nii$');
% F1 = nifti(files1);
% files2    = spm_select('FPList',datadir,'^wc2.*\.nii$');
% F2 = nifti(files2);
% 
% vaidx = 11:15;
% 
% for i=1:size(vaidx,2)
% F=cat(4,F1(vaidx(i)).dat,F2(vaidx(i)).dat);  
% [Y,~]=PatchCCAapply(model,F,sett); % Y is label map and P is prob
% label_val{i}=Y;
% end
% 
% files3= spm_select('FPList',datadir,'^w.*glm\.nii$');
% Nii=nifti(files3);
% Nii_1=Nii(11:15);
% 
% for j=1:size(label_val,2)
%     Z=label_val{j};
%     score_val{j}=ssim(double(Z(:,:,50)'), Nii_1(j).dat(:,:,50)');
% end
% 
% val_mean=mean([score_val{:}]);
%     
% %% testing
% 
% datadir = 'D:\Documents\Yu\Fusion-Challenge\testing-images'; % Edit accordingly 
% % datadir = 'D:\Documents\Yu\Fusion-Challenge\training-images';
% files  = spm_select('FPList',datadir,'^wc1.*\.nii$');
% v        = struct('view', {'',''});
% dat      = struct('view',  struct('image',{'',''}),...
%                   'jitter',[0 0 0]);
% 
% clear data
% data.code = [1 2];
% data.dat(1:size(files,1),1) = deal(dat);
% 
% for n=1:size(files,1)
% 
%     % Add the various GM and WM maps (assumed to be saved as wc1*.nii and wc2*.nii)
%     file_c1 = deblank(files(n,:)); % wc1 gray matter
%     [pth,nam,ext] = fileparts(file_c1);
%     file_c2  = fullfile(pth,[nam(1:2) '2' nam(4:end) ext]); %wc2 white matter
%     data.dat(n).view(1).image = cat(1,file_c1,file_c2); % concatenate
%     data.dat(n).jitter        = [0 0 0]; 
% 
%     % Include labels if present (assumed to be saved as w*_glm.nii)
%     file_glm = fullfile(pth,['w' nam(4:end) '_glm' ext]);
%     if exist(file_glm,'file')
%         data.dat(n).view(2).image = file_glm;
%         data.dat(n).jitter        = [2 2 2];                                                          
% 
%     end
%     
% end
% 
% for i=1:size(data.dat,1)
% F=cat(4,F1(i).dat,F2(i).dat);  
% [Y,~]=PatchCCAapply(model,F,sett); % Y is label map and P is prob
% label_test{i}=Y;
% end
% 
% files3= spm_select('FPList',datadir,'^w.*glm\.nii$');
% Nii=nifti(files3);
% 
% for j=1:size(files3,1)
%     Z=label_test{j};
%     score_test{j}=ssim(double(Z(:,:,50)'), Nii(j).dat(:,:,50)');
%     
% end
% 
% test_mean=mean([score_test{:}]);

