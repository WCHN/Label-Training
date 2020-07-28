function sett = PatchCCAsettings(settings)
% Default settings
def_sett = struct('K',25,'nit',5,'b0',1,'nu0',0,'v0',1,'d1',4,'nit0',8,'do_orth',true,'matname','','workers',0);
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
sett.nu0 = max(sett.nu0,sett.K*7);

