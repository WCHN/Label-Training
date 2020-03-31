function [Y,P]=PatchCCAapply(model,F,sett)
ind = 1;
if nargin>=3
    sett = PatchCCAsettings(sett);
else
    sett = PatchCCAsettings;
end
Z     = PatchCCAlatent(model,F,ind,sett);
if nargout>=1, Y     = PatchCCArecon(model,Z,2); end
if nargout>=2, [~,P] = PatchCCArecon(model,Z,1); end

