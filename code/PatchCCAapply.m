function [Y,P]=PatchCCAapply(model,F,sett)
% Run encoding and decoding on a target image
% FORMAT [Y,P] = PatchCCAapply(model,F,sett)
%     model - trained model (encoder in NN form)
%     F     - Categorical image data (3D/4D)
%     Sett  - settings (uses sett.nit0 & sett.nit)
%_______________________________________________________________________
% Copyright (C) 2019-2020 Wellcome Centre for Human Neuroimaging

ind = 1;
if nargin>=3
    sett = PatchCCAsettings(sett);
else
    sett = PatchCCAsettings;
end
Z     = PatchCCAlatent(model,F,ind,sett);
if nargout>=1, Y     = PatchCCArecon(model,Z,2); end
if nargout>=2, [~,P] = PatchCCArecon(model,Z,1); end

