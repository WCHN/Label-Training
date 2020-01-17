i1 = img(n).dat(:,:,64);
i2 = Nlab(n).dat(:,:,64);

figure(1);
subplot(121)
imagesc(i1)
axis image xy
subplot(122)
imagesc(i2)
axis image xy

labels = unique(i1);
for i=1:numel(labels)
    msk1 = i1 == labels(i);
    msk2 = i2 == labels(i);
    
    subplot(121)
    imagesc(msk1)
    axis image xy
    subplot(122)
    imagesc(msk2)
    axis image xy

    dice(msk1,msk2)
    
     pause(1)
end