function sett = PatchCCAsettings(settings)
% Default settings
%
%_______________________________________________________________________
% Copyright (C) 2019-2020 Wellcome Centre for Human Neuroimaging

def_sett = struct('K',25,'nit',5,'b0',1,'nu0',0,'v0',1,'d1',3,'nit0',10,'do_orth',true,'matname','','workers',0,'verb',0);
if nargin < 1 || isempty(settings)
    sett = def_sett;
else
    sett = settings;
    fnms = fieldnames(def_sett);
    for i=1:length(fnms)
        if ~isfield(sett,fnms{i})
            sett.(fnms{i}) = def_sett.(fnms{i});
        end
    end
end
if nargin >= 1 && isfield(settings,'nu0')
    sett.nu0 = settings.nu0;
else
    sett.nu0 = max(sett.nu0,sett.K*7-0.99);
end

