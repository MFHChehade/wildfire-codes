function pf_probe(case_name, out_json, enforce_q, alg, open_lines_json)
%PF_PROBE  Minimal PF wrapper to toggle Q limit enforcement & algorithm.
% Usage:
%   pf_probe('case118','out.json',1,1,'')             % enforce Q-lims, NR
%   pf_probe('case118','out.json',0,2,'opens.json')   % no Q-lims, FD, open lines
%
% open_lines_json (optional): JSON file with a field "open_lines": [line_ids]
% Lines are 1-based; STATUS column (branch(:,11)) is set to 0 for those.

    try, define_constants; catch, end

    out = struct();
    out.case = case_name;
    out.enforce_q = logical(enforce_q);
    out.alg = alg;

    % --- load case
    try
        mpc = loadcase(case_name);
    catch ME
        out.error = "loadcase_failed: " + string(ME.message);
        writejson(out_json, out); return;
    end

    % --- optionally open some lines
    try
        if exist('open_lines_json','var') && ~isempty(open_lines_json)
            if exist(open_lines_json, 'file')
                raw = fileread(open_lines_json);
                obj = jsondecode(raw);
                if isfield(obj, 'open_lines') && ~isempty(obj.open_lines)
                    ids = obj.open_lines(:)'; n = size(mpc.branch,1);
                    mpc.branch(:,11) = 1;
                    for k = 1:numel(ids)
                        lid = max(1, min(n, round(ids(k))));
                        mpc.branch(lid,11) = 0;
                    end
                    out.applied_open_lines = ids;
                end
            end
        end
    catch ME
        out.warn_open_lines = "open_lines_parse_failed: " + string(ME.message);
    end

    % --- options
    try
        mpopt = mpoption('verbose',0,'out.all',0);
        mpopt = mpoption(mpopt,'pf.tol',1e-8,'pf.nr.max_it',30);
        mpopt = mpoption(mpopt,'pf.enforce_q_lims',enforce_q);
        mpopt = mpoption(mpopt,'pf.alg', (alg==2)*2 + (alg~=2)*1);  % 1=NR, 2=FD
    catch ME
        out.error = "mpoption_failed: " + string(ME.message);
        writejson(out_json, out); return;
    end

    % --- run PF
    try
        r = runpf(mpc, mpopt);
        out.success = logical(r.success);
        if isfield(r,'et'), out.solve_time = r.et; end
        if isfield(r,'iterations'), out.iterations = r.iterations; end

        if isfield(r,'bus') && ~isempty(r.bus)
            VM = r.bus(:, VM);
            out.Vm = VM(:)'; out.Vm_min = min(VM); out.Vm_max = max(VM);
        end

        % crude Q-limit violation metric (how “far” beyond limits)
        if isfield(r,'gen') && ~isempty(r.gen)
            Qg = r.gen(:, QG); Qmin = r.gen(:, QMIN); Qmax = r.gen(:, QMAX);
            vio = max(Qg - Qmax, 0) + max(Qmin - Qg, 0);
            out.Q_violation_sum = sum(vio);
            out.Q_violating_gens = sum(vio > 0);
        end
    catch ME
        out.success = false;
        out.error = "runpf_failed: " + string(ME.message);
    end

    writejson(out_json, out);
end

function writejson(path, obj)
    fid = fopen(path,'w');
    if fid < 0, error('Cannot open %s for writing', path); end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', jsonencode(obj));
end
