function run_qmax_sweep(case_name, xi_json, limits_yml, plan_json, out_json, mode, step, n_steps)
% run_qmax_sweep
% Inputs:
%   case_name  : e.g., 'case118' (must exist in your MATLAB path / MATPOWER cases)
%   xi_json    : PSPS JSON path (used to set up outages/topology if your apply_xi helper exists)
%   limits_yml : limits file path (not required here, but kept for consistency)
%   plan_json  : plan JSON with {"corridor_actions":[{"action":"open","line":L, "name":optional},...]}
%   out_json   : output JSON path
%   mode       : 'abs' or 'pct'  (absolute Mvar decrement per step, or percentage factor per step)
%   step       : if 'abs' → e.g., 10 (Mvar); if 'pct' → e.g., 0.10 (=10% each step)
%   n_steps    : number of tightening steps
%
% Behavior:
%   - Runs DC once (records Pg; load shedding if your case supports it; otherwise returns NaN)
%   - Baseline AC PF with original QMIN/QMAX
%   - For k = 1..n_steps: cumulatively tighten QMAX and rerun PF
%   - Writes per-bus |V| across steps plus simple voltage penalties

    if nargin < 9, n_steps = 5; end
    if nargin < 8, step    = 10; end
    if nargin < 7, mode    = 'abs'; end

    % ---- Utilities ----
    function mpc = try_apply_xi(mpc, xi_path)
        % If you have a specific helper to apply PSPS topology, call it here.
        % Otherwise, assume xi carries nothing (no-op).
        try
            if exist('apply_xi','file')
                xi_struct = jsondecode(fileread(xi_path));
                mpc = apply_xi(mpc, xi_struct);
            end
        catch
            % no-op on error
        end
    end

    function mpc = apply_plan_open(mpc, plan_struct)
        % OPEN lines by branch index (1-based). Clamp to size.
        if ~isfield(plan_struct, 'corridor_actions'); return; end
        n_br = size(mpc.branch, 1);
        for i = 1:numel(plan_struct.corridor_actions)
            a = plan_struct.corridor_actions(i);
            if isfield(a, 'action') && strcmpi(a.action, 'open') && isfield(a, 'line')
                lid = max(1, min(n_br, round(double(a.line))));
                % MATPOWER: to open a line, set STATUS=0 (col 11)
                mpc.branch(lid, 11) = 0;
            end
        end
    end

    function pen = voltage_penalty(Vm, v_deadband, scale, metric)
        % Vm: per-bus voltage magnitudes
        over = max(abs(Vm - 1.0) - v_deadband, 0);
        if strcmpi(metric, 'L2')
            pen = scale * sum(over.^2);
        else
            pen = scale * sum(over);
        end
    end

    % ---- Load inputs ----
    plan = jsondecode(fileread(plan_json));
    % v* parameters are for reporting only (not gating here). Change if you keep a limits.yml parser.
    v_deadband = 0.03; v_scale = 30000; v_metric = 'L1';

    % ---- Load case & apply PSPS + plan ----
    define_constants;  % MATPOWER constants
    mpopt = mpoption('verbose', 0, 'out.all', 0);

    try
        mpc0 = loadcase(case_name);
    catch
        error('Failed to load MATPOWER case: %s', case_name);
    end

    % Optionally apply PSPS topology
    if ~isempty(xi_json) && isfile(xi_json)
        mpc0 = try_apply_xi(mpc0, xi_json);
    end

    % Apply OPEN actions
    mpc0 = apply_plan_open(mpc0, plan);

    % Save base Q limits
    QMIN = mpc0.gen(:, QMIN_);
    QMAX = mpc0.gen(:, QMAX_);

    % ---- DC run (optional details depend on your case modeling) ----
    dc = struct('converged', false, 'Pg', [], 'shed_total', NaN);
    try
        mpopt_dc = mpoption(mpopt, 'model', 'DC');
        rdc = rundcopf(mpc0, mpopt_dc);   % if OPF data present
        dc.converged = rdc.success == 1;
        if dc.converged
            dc.Pg = rdc.gen(:, PG);
            % If your case implements curtailable load as negative gens or extra fields,
            % compute a rough "shed" proxy; otherwise keep NaN.
            dc.shed_total = NaN;
        end
    catch
        % No DC OPF data → leave as NaN
    end

    % ---- Baseline AC PF ----
    base = struct('ac_ok', false, 'Vm', [], 'Vang', [], 'V_pen', Inf);
    try
        rac = runpf(mpc0, mpopt);
        base.ac_ok = (rac.success == 1);
        if base.ac_ok
            base.Vm   = abs(rac.bus(:, VM));
            base.Vang = rac.bus(:, VA);
            base.V_pen = voltage_penalty(base.Vm, v_deadband, v_scale, v_metric);
        end
    catch
        % leave as defaults
    end

    % ---- QMAX tightening sweep (cumulative) ----
    steps = struct('k', {}, 'qmax_factor', {}, 'qmax_delta', {}, 'ac_ok', {}, 'Vm', {}, 'V_pen', {});
    mpcK = mpc0; QMAXk = QMAX;  % start from baseline

    for k = 1:n_steps
        switch lower(mode)
            case 'abs'
                % decrement each Qmax by fixed Mvar, but never below Qmin+small
                QMAXk = max(QMIN + 1e-3, QMAXk - step);
                qmax_factor = NaN;    % not meaningful in abs mode
                qmax_delta  = step;   % Mvar decremented at this step
            case 'pct'
                % multiply the REMAINING headroom above Qmin by (1-step)
                % i.e., Qmax := Qmin + (Qmax - Qmin)*(1 - step)
                head = QMAXk - QMIN;
                head = max(0, head) .* (1 - step);
                QMAXk = QMIN + head;
                qmax_factor = (1 - step);
                qmax_delta  = NaN;
            otherwise
                error('mode must be ''abs'' or ''pct''');
        end

        mpcK.gen(:, QMAX_) = QMAXk;

        sk = struct('k', k, 'qmax_factor', qmax_factor, 'qmax_delta', qmax_delta, ...
                    'ac_ok', false, 'Vm', [], 'V_pen', Inf);
        try
            rak = runpf(mpcK, mpopt);
            sk.ac_ok = (rak.success == 1);
            if sk.ac_ok
                VmK = abs(rak.bus(:, VM));
                sk.Vm    = VmK;
                sk.V_pen = voltage_penalty(VmK, v_deadband, v_scale, v_metric);
            end
        catch
            % keep defaults
        end
        steps(end+1) = sk; %#ok<AGROW>
    end

    % ---- Write result JSON ----
    out = struct();
    out.case_name = case_name;
    out.plan_text = plan_to_text(plan);
    out.dc = dc;
    out.base = base;
    out.steps = steps;

    if ~isempty(out_json)
        fid = fopen(out_json, 'w');
        fprintf(fid, '%s', jsonencode(out));
        fclose(fid);
    end
end

function s = plan_to_text(plan)
    s = "";
    if isfield(plan, 'corridor_actions')
        parts = strings(1, numel(plan.corridor_actions));
        for i = 1:numel(plan.corridor_actions)
            a = plan.corridor_actions(i);
            if isfield(a,'action') && strcmpi(a.action,'open') && isfield(a,'line')
                lid = round(double(a.line));
                nm = ""; if isfield(a,'name') && ~isempty(a.name), nm = string(a.name)+":"; end
                parts(i) = "open(" + nm + string(lid) + ")";
            end
        end
        s = strjoin(parts, "; ");
    end
end
