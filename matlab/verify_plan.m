function verify_plan(case_name, plan_path, xi_path, limits_path, out_path)
% VERIFY_PLAN
% Build (z, delta_g=0) from a plan, solve DC OPF-with-shedding (Png optimized),
% and write JSON {feasible, J, shed_MW, notes}.
%
% plan.corridor_actions items can be:
%   { "action":"open", "line": <id> }                   % open specific line
%   { "action":"open", "name":"Sk", "line": <id> }      % open specific line in corridor Sk
%   { "action":"open", "name":"Sk" }                    % pick first eligible line in Sk
%
% If corridor_actions == [] (empty), do nothing gracefully.

clc; yalmip('clear');

%% --- Load case ---
switch lower(case_name)
    case 'case118'
        mpopt = mpoption('verbose', 0, 'out.all', 0);
        mpc   = rundcpf('case118', mpopt);
    otherwise
        error('Unsupported case: %s', case_name);
end
n_bus  = size(mpc.bus,1);
n_line = size(mpc.branch,1);

%% --- Load configs / inputs ---
limits   = read_limits_yaml(limits_path);
corrmap  = jsondecode(fileread('config/corridor_map.json'));
plan     = jsondecode(fileread(plan_path));
xi_raw   = jsondecode(fileread(xi_path));

% Normalize xi -> {0,1}^n_line
xi = normalize_xi(xi_raw, n_line);

%% --- Build (z, delta_g) ---
z = ones(n_line,1);
z(xi==0) = 0;  % PSPS forced-open

budget       = value_or_default(limits,'budget',3);
toggles_used = 0;

% Normalize corridor_actions into a struct array (or struct([]))
ca_list = normalize_actions(plan, 'corridor_actions');

for a = 1:numel(ca_list)
    if toggles_used >= budget, break; end
    act = to_str_scalar(ca_list(a), 'action');
    if act ~= "open", continue; end

    % 1) If an explicit line id is provided, try to open it (if eligible)
    lid = get_numeric_field(ca_list(a), 'line', NaN);
    if isfinite(lid)
        lid = round(lid);
        if lid>=1 && lid<=n_line && xi(lid)==1 && z(lid)==1
            z(lid) = 0; toggles_used = toggles_used + 1; 
            continue;
        end
    end

    % 2) Otherwise, if a corridor name exists, pick the first eligible line inside it
    S = to_str_scalar(ca_list(a), 'name');
    if S ~= "" && isfield(corrmap, S)
        ids = corrmap.(S); ids = ids(:).';
        pick = [];
        for id = ids
            id = round(id);
            if id>=1 && id<=n_line && xi(id)==1 && z(id)==1
                pick = id; break;
            end
        end
        if ~isempty(pick)
            z(pick) = 0; toggles_used = toggles_used + 1;
        end
    end
end

% No redispatch in this project stage
delta_g = zeros(n_bus,1);

%% --- Verify (optimize Png, theta, P, Ps given z, xi, delta_g=0) ---
try
    [feasible,J,shedMW] = run_dc_verifier(mpc, z, delta_g, xi, limits);
catch ME
    warning("Verifier error: %s", ME.message);
    feasible = false; J = 1e9; shedMW = NaN;
end

%% --- Write result ---
result = struct();
result.feasible = logical(feasible);
result.J        = J;
result.shed_MW  = shedMW;
result.notes    = ternary(feasible, "OK", "INFEASIBLE");

fid = fopen(out_path,'w');
assert(fid>0, 'Cannot open output path: %s', out_path);
fprintf(fid,'%s', jsonencode(result));
fclose(fid);
end

% =========================
% ===== Helper Routines ===
% =========================

function ca = normalize_actions(plan, fieldname)
% Return a struct array for plan.(fieldname). Accept:
%   - missing: struct([])
%   - []      : struct([])
%   - struct  : return as 1x1
%   - struct array: return as-is
% Anything else -> struct([])

ca = struct([]);
if ~isstruct(plan) || ~isfield(plan, fieldname) || isempty(plan.(fieldname))
    return;
end
v = plan.(fieldname);
if isstruct(v)
    ca = v;  % already struct or struct array
else
    % jsondecode([]) -> [] (double), or other non-struct -> ignore
    ca = struct([]);
end
end

function xi = normalize_xi(xi_raw, n_line)
% Convert xi inputs into a {0,1} mask of length n_line.

xi = ones(n_line,1); % default all available

if isstruct(xi_raw)
    if isfield(xi_raw,'xi')
        v = double(xi_raw.xi(:) ~= 0);
        xi = fit_mask_len(v, n_line);
    elseif isfield(xi_raw,'forced_open')
        idx = round(xi_raw.forced_open(:));
        idx = idx(idx>=1 & idx<=n_line);
        xi = ones(n_line,1); xi(idx)=0;
    else
        error('xi struct must include "xi" or "forced_open".');
    end
elseif isnumeric(xi_raw) || islogical(xi_raw)
    v = xi_raw(:);
    u = unique(v(~isnan(v)));
    if all(ismember(u',[0 1]))
        v  = double(v~=0);
        xi = fit_mask_len(v, n_line);
    elseif all(v>=1 & v<=n_line) && numel(v)<n_line
        xi = ones(n_line,1); xi(round(v))=0;
    else
        error('Unsupported xi numeric format (len=%d). Provide 0/1 mask or index list.', numel(v));
    end
else
    error('Unsupported xi JSON format.');
end
end

function xi = fit_mask_len(v, n_line)
if numel(v) < n_line
    xi = ones(n_line,1);
    xi(1:numel(v)) = v;
elseif numel(v) > n_line
    xi = v(1:n_line);
else
    xi = v;
end
end

function L = read_limits_yaml(path)
txt = fileread(path);
L = struct();
L.budget         = str2double_maybe(extract_first(txt, 'budget:\s*([0-9\.\-eE]+)'));
L.redispatch_cap = str2double_maybe(extract_first(txt, 'redispatch_cap:\s*([0-9\.\-eE]+)'));
L.gamma          = str2double_maybe(extract_first(txt, 'gamma:\s*([0-9\.\-eE]+)'));
L.lambda         = str2double_maybe(extract_first(txt, 'lambda:\s*([0-9\.\-eE]+)'));
if isnan(L.gamma),  L.gamma  = 100; end
if isnan(L.lambda), L.lambda = 0;   end
end

function val = extract_first(txt, pat)
m = regexp(txt, pat, 'tokens', 'once');
if isempty(m), val = ''; else, val = m{1}; end
end

function x = str2double_maybe(s)
if isempty(s), x = NaN; else, x = str2double(s); end
end

function s = to_str_scalar(st, field)
if ~isfield(st, field) || isempty(st.(field)), s = ""; return; end
v = st.(field);
if isstring(v), s = v(1); return; end
if ischar(v),   s = string(v); return; end
s = string(v);
end

function v = get_numeric_field(st, field, def)
if ~isfield(st, field) || isempty(st.(field)), v = def; return; end
v = st.(field);
if ~isnumeric(v) || ~isscalar(v) || ~isfinite(v), v = def; end
end

function out = value_or_default(S, key, def)
if isfield(S, key) && ~isempty(S.(key)) && isfinite(S.(key))
    out = S.(key);
else
    out = def;
end
end

function y = ternary(cond, a, b)
if cond, y = a; else, y = b; end
end

% ===== DC OPF-with-shedding (LP; Png optimized) =====
function [feasible,J,shedMW] = run_dc_verifier(mpc,z,delta_g,xi,limits)
yalmip('clear');
options = sdpsettings('verbose', 0);

nb = size(mpc.bus,1);
nl = size(mpc.branch,1);

% costs
c = zeros(nb,1);
c(10)=0.217; c(12)=1.052; c(25)=0.434; c(26)=0.308; c(31)=5.882; c(46)=3.448;
c(49)=0.467; c(54)=1.724; c(59)=0.606; c(61)=0.588; c(65)=0.2493; c(66)=0.2487;
c(69)=0.1897; c(80)=0.205; c(87)=7.142; c(92)=10; c(100)=0.381; c(103)=2; c(111)=2.173;

Pd = mpc.bus(:,3); Pd(90)=440;

Pmax = 220*ones(nl,1);
b440 = [3 21 31 33 50 96 98 99 90 93 94 97 107 108 116 123 137 163];
b660 = [38 36 51 138 140];
Pmax(b440)=440; Pmax(b660)=660; Pmax(7)=1100; Pmax(9)=1100; Pmax(8)=880;

Bline = 1./mpc.branch(:,4);
A = zeros(nb,nl);
for i=1:nb
    for j=1:nl
        if mpc.branch(j,1)==i, A(i,j)= 1; end
        if mpc.branch(j,2)==i, A(i,j)=-1; end
    end
end
M = zeros(nl,nb);
for j=1:nl
    i1=mpc.branch(j,1); i2=mpc.branch(j,2);
    M(j,i1)= Bline(j); M(j,i2)=-Bline(j);
end

Png_min = zeros(nb,1);
Png_max = zeros(nb,1);
Png_max(10)=550; Png_max(12)=185; Png_max(25)=320; Png_max(26)=414; Png_max(31)=107;
Png_max(46)=119; Png_max(49)=304; Png_max(54)=148; Png_max(59)=255; Png_max(61)=260;
Png_max(65)=491; Png_max(66)=492; Png_max(69)=805.2; Png_max(80)=577; Png_max(87)=104;
Png_max(92)=100; Png_max(100)=352; Png_max(103)=140; Png_max(111)=136;

theta = sdpvar(nb,1);
P     = sdpvar(nl,1);
Ps    = sdpvar(nb,1);
Png   = sdpvar(nb,1);

gamma  = value_or_default(limits,'gamma',100);
lambda = value_or_default(limits,'lambda',0);
obj = c'*Png + gamma*sum(Ps) + lambda*sum(abs(delta_g));

con = {};
con{end+1} = -ones(nb,1) <= theta <= ones(nb,1);
con{end+1} = theta(1) == 0;
con{end+1} = Png_min <= Png + delta_g <= Png_max;
con{end+1} = A*P == (Png + delta_g) - (Pd - Ps);
con{end+1} = Ps >= 0;

gate = (xi(:).*z(:));
con{end+1} = -Pmax.*gate <= P <= Pmax.*gate;

Moff=1000; scale=100;
for j=1:nl
    con{end+1} =  scale*M(j,:)*theta - P(j) + (1-gate(j))*Moff >= 0;
    con{end+1} =  scale*M(j,:)*theta - P(j) - (1-gate(j))*Moff <= 0;
end

sol = optimize([con{:}], obj, options);
feasible = (sol.problem==0);
if feasible
    J = value(obj);
    shedMW = sum(value(Ps));
else
    J = 1e9; shedMW = NaN;
end
end
