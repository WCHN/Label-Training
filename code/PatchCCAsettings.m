function sett = PatchCCAsettings(settings)
% Default settings
def_sett = struct('K',25,'nit',5,'b0',1,'nu0',1500,'v0',2,'d1',3,'nit0',10,'do_orth',false,'matname','','workers',0);
if nargin < 1 || isempty(settings)
    sett = def_sett;
else
    sett = settings;  
    fnms = fieldnames(def_sett); % cell array with field names
    for i=1:length(fnms)
        if ~isfield(sett,fnms{i})
            sett.(fnms{i}) = def_sett.(fnms{i});
        end
    end
end
sett.nu0 = max(sett.nu0,sett.K*7);

