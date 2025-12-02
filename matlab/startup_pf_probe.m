function startup_pf_probe(matlab_dir, matpower_dir)
% Add your repo matlab/ and MATPOWER to path in batch sessions.
    addpath(genpath(matlab_dir));
    if nargin >= 2 && ~isempty(matpower_dir)
        addpath(genpath(matpower_dir));
    end
    try, define_constants; catch, end
    % Optional sanity prints
    which pf_probe
    which loadcase
end
