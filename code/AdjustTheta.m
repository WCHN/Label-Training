function theta1 = AdjustTheta(theta,mat)
theta1 = theta;
M      = inv(mat);
theta1(:,:,:,1) = M(1,1)*theta(:,:,:,1)+M(1,2)*theta(:,:,:,2)+M(1,3)*theta(:,:,:,3)+M(1,4);
theta1(:,:,:,2) = M(2,1)*theta(:,:,:,1)+M(2,2)*theta(:,:,:,2)+M(2,3)*theta(:,:,:,3)+M(2,4);
theta1(:,:,:,3) = M(3,1)*theta(:,:,:,1)+M(3,2)*theta(:,:,:,2)+M(3,3)*theta(:,:,:,3)+M(3,4);

