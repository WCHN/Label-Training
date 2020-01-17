function o = Overlap(Y0,Y)
% A slightly ad hoc overlap measure - to be formalised later
o = sum(sum(sum(Y==Y0 & Y~=0)))/sum(sum(sum(Y0~=0 | Y~=0)));
fprintf('%g\n', o*100);
