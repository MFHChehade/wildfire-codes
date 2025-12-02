
function qmax_sweep_minimal(case_name, scales, v_deadband, v_scale, out_json)
%QMAX_SWEEP_MINIMAL Sweep generator Qmax → AC PF voltages & penalty.
%   qmax_sweep_minimal('case118',[1 0.95 0.9],0.03,30000,'out.json')
% Requires: MATPOWER on MATLAB path.
define_constants;  % MATPOWER constants (QMAX, QG, VM, etc.)

mpc = loadcase(case_name);
orig_Qmax = mpc.gen(:, QMAX);

% ---- Build mpopt and enforce Q limits (version-safe) ----
mpopt = mpoption('verbose', 0, 'out.all', 0);
try
    % MATPOWER >= 7 style
    mpopt = mpoption(mpopt, 'pf.enforce_q_lims', 1);
catch
    % Older MATPOWER: legacy option
    mpopt = mpoption(mpopt, 'ENFORCE_Q_LIMS', 1);
end

results = struct();
steps   = [];

% ---------- Baseline PF (scale = 1.0) ----------
try
    r0 = runpf(mpc, mpopt);
    base_ok = (r0.success == 1);
    if base_ok
        Vm0    = r0.bus(:, VM);
        Qg0    = r0.gen(:, QG);
        V_pen0 = voltage_penalty(Vm0, v_deadband, v_scale);
    else
        Vm0    = [];
        Qg0    = [];
        V_pen0 = NaN;
    end
catch ME
    base_ok = false;
    Vm0     = [];
    Qg0     = [];
    V_pen0  = NaN;
    warning('Baseline PF failed: %s', ME.message);
end

results.base = struct( ...
    'ac_ok', base_ok, ...
    'V_pen', V_pen0, ...
    'Vm',    Vm0(:)', ...
    'Qg',    Qg0(:)', ...
    'Qmax',  orig_Qmax(:)' ...
);

% ---------- Sweep Qmax scales ----------
for k = 1:numel(scales)
    s = scales(k);
    m = mpc;
    m.gen(:, QMAX) = min(orig_Qmax, orig_Qmax * s);  % scaled Qmax
    cur_Qmax = m.gen(:, QMAX);

    try
        rk = runpf(m, mpopt);
        ok = (rk.success == 1);
        if ok
            Vm = rk.bus(:, VM);
            Qg = rk.gen(:, QG);
            Vp = voltage_penalty(Vm, v_deadband, v_scale);
        else
            Vm = [];
            Qg = [];
            Vp = NaN;
        end
    catch ME
        ok = false;
        Vm = [];
        Qg = [];
        Vp = NaN;
        warning('PF failed at scale=%.4f: %s', s, ME.message);
    end

    steps = [steps, struct( ...
        'k',      k, ...
        'scale',  s, ...
        'ac_ok',  ok, ...
        'V_pen',  Vp, ...
        'Vm',     Vm(:)', ...
        'Qg',     Qg(:)', ...
        'Qmax',   cur_Qmax(:)' ...
    )];
end

results.steps = steps;

% ---------- Write JSON ----------
try
    txt = jsonencode(results);
    fid = fopen(out_json, 'w'); fwrite(fid, txt); fclose(fid);
catch ME
    error('Failed to write JSON: %s', ME.message);
end
end

function Vp = voltage_penalty(Vm, v_deadband, v_scale)
    if isempty(Vm)
        Vp = NaN; return;
    end
    dev = abs(Vm) - 1.0 - v_deadband;
    dev(dev < 0) = 0;
    Vp = v_scale * sum(dev);
end
