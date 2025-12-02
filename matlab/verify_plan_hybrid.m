function verify_plan_hybrid(case_name, plan_json, xi_json, limits_yml, out_json)
%VERIFY_PLAN
% Hybrid verifier for a PSPS + switching plan.
% 1) Builds branch status from xi + plan.
% 2) Runs DC POWER FLOW (rundcpf) as a feasibility gate.
% 3) Runs AC PF (runpf) and computes voltage penalty.
%
% Writes JSON:
%   {
%     feasible,         % overall feasibility
%     J_dc,             % DC "cost" (here 0 if DC PF succeeds, big penalty otherwise)
%     V_pen,            % voltage penalty from AC
%     J_ac,             % same as V_pen
%     J,                % alpha * J_dc + beta * J_ac
%     ac_ok,            % whether AC PF converged
%     notes             % debug string
%   }

    % ---------- IO ----------
    plan = jsondecode(fileread(plan_json));
    xi   = jsondecode(fileread(xi_json));
    xi   = double(xi(:) ~= 0);   % 0/1 column vector

    % Limits (YAML/JSON) with safe fallbacks
    limits          = load_limits_safe(limits_yml);
    alpha           = value_or_default(limits,'alpha',1.0);
    beta            = value_or_default(limits,'beta',1.0);
    v_deadband      = value_or_default(limits,'v_deadband',0.03);
    v_metric        = value_or_default(limits,'voltage_metric','L1');
    v_scale         = value_or_default(limits,'voltage_scale',30000);
    penalty_dc_fail = value_or_default(limits,'penalty_dc_fail',1e9);
    penalty_ac_fail = value_or_default(limits,'penalty_ac_fail',1e6);

    % ---------- Load base case ----------
    mpc = loadcase(case_name);

    nline = size(mpc.branch,1);
    if numel(xi) ~= nline
        error('xi length (%d) does not match case branch count (%d)', numel(xi), nline);
    end

    % Base PSPS mask: 1 = in-service, 0 = out-of-service
    mpc.branch(:,11) = 1;
    mpc.branch(xi==0,11) = 0;

    % ---- Normalize corridor_actions and apply opens ----
    acts = normalize_actions(plan);
    for k = 1:numel(acts)
        a = acts(k);
        if ~strcmpi(a.action,'open')
            continue;
        end
        lid = a.line;
        if ~(isfinite(lid) && lid >= 1 && lid <= nline)
            continue;
        end
        mpc.branch(lid,11) = 0;
    end

    % =====================================================================
    % Step 1: DC POWER FLOW via MATPOWER (feasibility + J_dc)
    % =====================================================================
    try
        mpopt_dc = mpoption('VERBOSE',0,'OUT_ALL',0);
        dcres = rundcpf(mpc, mpopt_dc);   % DC PF, NOT OPF

        if dcres.success
            dc_feasible = true;
            J_dc        = 0;              % dummy DC cost (no explicit DC OPF)
            note_dc     = 'DC PF success';
        else
            dc_feasible = false;
            J_dc        = penalty_dc_fail;
            note_dc     = 'DC PF failed';
        end
    catch ME
        dc_feasible = false;
        J_dc        = penalty_dc_fail;
        note_dc     = ['DC PF exception: ' ME.message];
    end

    % If DC fails badly, you can choose to bail out here.
    if ~dc_feasible
        out          = struct();
        out.feasible = false;
        out.J_dc     = J_dc;
        out.V_pen    = NaN;
        out.J_ac     = NaN;
        out.J        = alpha * J_dc + beta * penalty_ac_fail;
        out.ac_ok    = false;
        out.notes    = note_dc;
        write_json(out_json, out);
        return
    end

    % =====================================================================
    % Step 2: AC POWER FLOW and voltage penalty
    % =====================================================================
    try
        mpopt_ac = mpoption('VERBOSE', 0, 'OUT_ALL', 0, 'ENFORCE_Q_LIMS', 1);
        acres = runpf(mpc, mpopt_ac);
        if ~acres.success
            ac_ok   = false;
            V_pen   = NaN;
            J_ac    = penalty_ac_fail;
            note_ac = 'AC PF failed';
        else
            ac_ok = true;
            V = abs(acres.bus(:,8));           % VM column
            dev = max(abs(V - 1.0) - v_deadband, 0);
            switch upper(v_metric)
                case 'L1'
                    raw = sum(dev);
                case 'L2'
                    raw = sum(dev.^2);
                otherwise
                    raw = sum(dev);
            end
            V_pen   = v_scale * raw;
            J_ac    = V_pen;                   % define J_ac as voltage penalty
            note_ac = 'AC OK';
        end
    catch ME
        ac_ok   = false;
        V_pen   = NaN;
        J_ac    = penalty_ac_fail;
        note_ac = ['AC exception: ' ME.message];
    end

    % =====================================================================
    % Step 3: Hybrid objective and output
    % =====================================================================
    J_hyb = alpha * J_dc + beta * J_ac;

    out          = struct();
    out.feasible = true;
    out.J_dc     = J_dc;
    out.V_pen    = V_pen;
    out.J_ac     = J_ac;
    out.J        = J_hyb;
    out.ac_ok    = ac_ok;
    out.notes    = [note_dc ' | ' note_ac];
    write_json(out_json, out);
end

% ===== Normalization helpers ==================================================
function acts = normalize_actions(plan)
% Return struct array with uniform fields: action,line,name

    acts_raw = [];
    if isstruct(plan) && isfield(plan,'corridor_actions')
        acts_raw = plan.corridor_actions;
    end

    out = struct('action',{},'line',{},'name',{});

    % 1) struct array already
    if isstruct(acts_raw)
        for i = 1:numel(acts_raw)
            a = normalize_one_act(acts_raw(i));
            if ~isempty(a)
                out(end+1) = a; %#ok<AGROW>
            end
        end
        acts = out;
        if ~isempty(acts), return; end
    end

    % 2) cell array (possibly mixed/mismatched)
    if iscell(acts_raw)
        for i = 1:numel(acts_raw)
            ai = acts_raw{i};
            if isstruct(ai)
                a = normalize_one_act(ai);
            elseif ischar(ai)
                a = parse_actions_text_to_struct(ai);
            else
                a = struct([]);
            end
            if ~isempty(a)
                out(end+1) = a; %#ok<AGROW>
            end
        end
        acts = out;
        if ~isempty(acts), return; end
    end

    % 3) raw text
    if ischar(acts_raw)
        acts = parse_actions_text_to_struct(acts_raw);
        if ~isempty(acts), return; end
    end

    % 4) or if plan is raw text itself (no corridor_actions key)
    if isstruct(plan) && isfield(plan,'actions_text')
        acts = parse_actions_text_to_struct(plan.actions_text);
        if ~isempty(acts), return; end
    end

    % 5) nothing usable
    acts = struct('action',{},'line',{},'name',{});
end

function aout = normalize_one_act(a)
% Map any struct with fields {action?, line, name?} into canonical schema.
    aout = struct([]);
    if ~isstruct(a)
        return;
    end

    % action
    act = 'open';
    if isfield(a,'action')
        act_val = a.action;
        if ischar(act_val)
            act = act_val;
        end
        if isempty(act)
            act = 'open';
        end
    end

    % line
    lid = NaN;
    if isfield(a,'line')
        lv = a.line;
        if isnumeric(lv)
            lid = double(lv);
        elseif ischar(lv)
            lid = str2double(lv);
        end
    end
    if ~isfinite(lid)
        return;
    end

    % name
    name = '';
    if isfield(a,'name') && ~isempty(a.name)
        name_val = a.name;
        if ischar(name_val)
            name = upper(name_val);
        else
            name = upper(char(name_val));
        end
    end

    aout = struct('action',act,'line',lid,'name',name);
end

function acts = parse_actions_text_to_struct(txt)
% Parse tokens like 'open(S6:135); open(131)' to a struct array.
    txt    = char(txt);
    pat    = 'open\s*\(\s*(?:([sS]\d+)\s*:\s*)?(\d+)\s*\)';
    tokens = regexp(txt, pat, 'tokens');
    acts   = struct('action',{},'line',{},'name',{});
    for i = 1:numel(tokens)
        t = tokens{i};
        cname = '';
        if numel(t) >= 1 && ~isempty(t{1})
            cname = upper(t{1});
        end
        lid = str2double(t{end});
        if ~isfinite(lid)
            continue;
        end
        a = struct('action','open','line',lid,'name',cname);
        acts(end+1) = a; %#ok<AGROW>
    end
end

% ===== Generic helpers ========================================================
function v = value_or_default(s, key, def)
    if isstruct(s) && isfield(s, key)
        v = s.(key);
        if isempty(v)
            v = def;
        end
    else
        v = def;
    end
end

function s = load_limits_safe(yml_path)
    s = struct();
    try
        if exist('yamlread','file') == 2
            s = yamlread(yml_path);
        elseif exist('ReadYaml','file') == 2
            s = ReadYaml(yml_path);
        else
            txt = fileread(yml_path);
            s   = jsondecode(txt);
        end
    catch
        % keep defaults
    end
end

function write_json(path, obj)
    fid = fopen(path,'w');
    fprintf(fid, '%s', jsonencode(obj));
    fclose(fid);
end
