function f1 = WarpLabels(f0,theta)

dm0 = size(theta);
if numel(dm0)~=4,    error('Inappropriate deformation.'); end
if ~isa(f0,'uint8'), error('Labels must be uint8.'); end

f1  = zeros(dm0(1:3),class(f0));
p1  = zeros(size(f1),'single');
for j=0:255
    if any(f0(:)==j)
        g0        = single(f0==j);
        g0        = convn(g0,reshape([0.25 0.5 0.25],[3,1,1]),'same');
        g0        = convn(g0,reshape([0.25 0.5 0.25],[1,3,1]),'same');
        g0        = convn(g0,reshape([0.25 0.5 0.25],[1,1,3]),'same');
        tmp       = spm_diffeo('pull',g0,theta);
        msk       = (tmp>p1);
        p1(msk)   = tmp(msk);
        f1(msk)   = j;
        fprintf('.');
    end
end
fprintf('\n');