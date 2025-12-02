function run_ground_truth(case_name, xi_path, limits_path, out_path, mode)
% RUN_GROUND_TRUTH  Baseline vs MILP-opt ground truth scorer (open-only).
% Inputs:
%   case_name   : 'case118'
%   xi_path     : io/psps_*.json  (0/1 mask OR {"forced_open":[...]} OR {"xi":[...]})
%   limits_path : config/limits.yml
%   out_path    : where to write JSON result
%   mode        : 'baseline' or 'mip_opt'
%
% Output JSON (superset of what you already use):
%   {
%     "feasible": bool,
%     "J": double,
%     "shed_MW": double,
%     "notes": "OK|INFEASIBLE",
%     "opt_switches": [ ... ],                 % (only in mip_opt)
%     "opt_plan": {"corridor_actions":[...]}   % (only in mip_opt)
%   }

clc; yalmip('clear');

%% --- Load case ---
switch lower(case_name)
    case 'case118'
        mpc = rundcpf('case118');
    otherwise
        error('Unsupported case: %s', case_name);
end
nb = size(mpc.bus,1); nl = size(mpc.branch,1);

%% --- Inputs ---
xi_raw  = jsondecode(fileread(xi_path));
xi      = normalize_xi(xi_raw, nl);   % {0,1}^nl
limits  = read_limits_yaml(limits_path);
corrmap = load_corridor_map('config/corridor_map.json');

%% --- Model constants (same as your verifier) ---
% costs
c = zeros(nb,1);
c(10)=0.217; c(12)=1.052; c(25)=0.434; c(26)=0.308; c(31)=5.882; c(46)=3.448;
c(49)=0.467; c(54)=1.724; c(59)=0.606; c(61)=0.588; c(65)=0.2493; c(66)=0.2487;
c(69)=0.1897; c(80)=0.205; c(87)=7.142; c(92)=10; c(100)=0.381; c(103)=2; c(111)=2.173;

Pd = mpc.bus(:,3); Pd(90) = 440;

Pmax = 220*ones(nl,1);
Pmax([3 21 31 33 50 96 98 99 90 93 94 97 107 108 116 123 137 163]) = 440;
Pmax([38 36 51 138 140]) = 660;
Pmax(7) = 1100; Pmax(9) = 1100; Pmax(8) = 880;

% DC incidence and M
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
    i1 = mpc.branch(j,1); i2 = mpc.branch(j,2);
    M(j,i1) =  Bline(j); M(j,i2) = -Bline(j);
end

Png_min = zeros(nb,1); Png_max = zeros(nb,1);
Png_max(10)=550; Png_max(12)=185; Png_max(25)=320; Png_max(26)=414; Png_max(31)=107;
Png_max(46)=119; Png_max(49)=304; Png_max(54)=148; Png_max(59)=255; Png_max(61)=260;
Png_max(65)=491; Png_max(66)=492; Png_max(69)=805.2; Png_max(80)=577; Png_max(87)=104;
Png_max(92)=100; Png_max(100)=352; Png_max(103)=140; Png_max(111)=136;

gamma  = value_or_default(limits,'gamma',100);
budget = value_or_default(limits,'budget',3);

%% --- Variables, objective, constraints ---
yalmip('clear'); options = sdpsettings('verbose', 0);

theta = sdpvar(nb,1);
P     = sdpvar(nl,1);
Ps    = sdpvar(nb,1);
Png   = sdpvar(nb,1);

obj   = c'*Png + gamma*sum(Ps);
con   = {};
con{end+1} = -ones(nb,1) <= theta <= ones(nb,1);
con{end+1} = theta(1) == 0;
con{end+1} = Png_min <= Png <= Png_max;
con{end+1} = A*P == (Png) - (Pd - Ps);
con{end+1} = Ps >= 0;

Moff=1000; scale=100;

switch lower(mode)
    case 'baseline'
        % Fix gate to xi (no extra toggles)
        gate = xi(:);
        con{end+1} = -Pmax.*gate <= P <= Pmax.*gate;
        for j=1:nl
            con{end+1} =  scale*M(j,:)*theta - P(j) + (1-gate(j))*Moff >= 0;
            con{end+1} =  scale*M(j,:)*theta - P(j) - (1-gate(j))*Moff <= 0;
        end
        sol = optimize([con{:}], obj, options);
        feasible = (sol.problem==0);
        J = feasible * value(obj) + ~feasible * 1e9;
        shedMW = feasible * sum(value(Ps)) + ~feasible * NaN;
        result = struct('feasible', logical(feasible), 'J', J, 'shed_MW', shedMW, ...
                        'notes', ternary(feasible,"OK","INFEASIBLE"));

    case 'mip_opt'
        % Binary topology z, respect PSPS and a toggle budget beyond PSPS
        z = binvar(nl,1);
        % PSPS-forced lines must be open:
        for j=1:nl
            if xi(j)==0
                con{end+1} = (z(j) == 0);
            end
        end
        % extra opens beyond PSPS bounded by budget:
        con{end+1} = sum(xi.*(1 - z)) <= budget;

        gate = xi(:).*z;                               % active only if both allow/closed
        con{end+1} = -Pmax.*gate <= P <= Pmax.*gate;
        for j=1:nl
            con{end+1} =  scale*M(j,:)*theta - P(j) + (1-gate(j))*Moff >= 0;
            con{end+1} =  scale*M(j,:)*theta - P(j) - (1-gate(j))*Moff <= 0;
        end
        sol = optimize([con{:}], obj, options);
        feasible = (sol.problem==0);
        if feasible
            J = value(obj); shedMW = sum(value(Ps));
            z_star = round(value(z));
            % operator toggles = lines opened by operator (xi=1 & z=0)
            op_toggles = find(xi(:)==1 & z_star(:)==0);
            % map to corridors for a label
            ca = corridor_actions_from_toggles(op_toggles, corrmap);
            result = struct('feasible',true,'J',J,'shed_MW',shedMW,'notes',"OK", ...
                            'opt_switches', op_toggles(:).', ...
                            'opt_plan', struct('corridor_actions', ca));
        else
            result = struct('feasible',false,'J',1e9,'shed_MW',NaN,'notes',"INFEASIBLE", ...
                            'opt_switches', [], 'opt_plan', struct('corridor_actions',[]));
        end

    otherwise
        error('Unsupported mode: %s', mode);
end

fid=fopen(out_path,'w'); fprintf(fid,'%s',jsonencode(result)); fclose(fid);

end

% ---------- helpers ----------
function xi = normalize_xi(xi_raw, n_line)
xi = ones(n_line,1);
if isstruct(xi_raw)
    if isfield(xi_raw,'xi')
        v = double(xi_raw.xi(:)~=0);
        xi(1:min(end,numel(v))) = v(1:min(end,numel(v)));
    elseif isfield(xi_raw,'forced_open')
        idx = round(xi_raw.forced_open(:));
        idx = idx(idx>=1 & idx<=n_line);
        xi(idx)=0;
    else
        error('xi struct must include "xi" or "forced_open".');
    end
elseif isnumeric(xi_raw) || islogical(xi_raw)
    v = xi_raw(:);
    if all(ismember(unique(v)',[0 1]))
        v = double(v~=0);
        xi(1:min(end,numel(v))) = v(1:min(end,numel(v)));
    else
        % treat as index list
        v = round(v);
        v = v(v>=1 & v<=n_line);
        xi = ones(n_line,1); xi(v)=0;
    end
else
    error('Unsupported xi format');
end
end

function L = read_limits_yaml(path)
txt = fileread(path);
L = struct();
L.budget = str2double_maybe(extract_first(txt,'budget:\s*([0-9\.\-eE]+)'));
L.gamma  = str2double_maybe(extract_first(txt,'gamma:\s*([0-9\.\-eE]+)'));
if isnan(L.budget), L.budget=3; end
if isnan(L.gamma),  L.gamma=100; end
end

function m = extract_first(txt, pat)
t = regexp(txt, pat, 'tokens', 'once'); if isempty(t), m=''; else, m=t{1}; end
end
function x = str2double_maybe(s); if isempty(s), x=NaN; else, x=str2double(s); end; end
function y = ternary(c,a,b); if c, y=a; else, y=b; end; end

function cmap = load_corridor_map(path)
cmap = jsondecode(fileread(path));
end

function ca = corridor_actions_from_toggles(toggles, corrmap)
% Build [{"name": "...","action":"open","line": id}, ...]
ca = struct('name', {}, 'action', {}, 'line', {});
names = fieldnames(corrmap);
for k=1:numel(toggles)
    id = toggles(k);
    cname = "";
    for i=1:numel(names)
        arr = corrmap.(names{i});
        if any(arr(:)==id)
            cname = string(names{i}); break;
        end
    end
    if cname==""
        cname = "UNK";
    end
    ca(end+1).name   = cname; %#ok<AGROW>
    ca(end).action   = "open";
    ca(end).line     = id;
end
end

function out = value_or_default(S, key, def)
% Returns S.(key) if it exists and is finite, otherwise def.
if isstruct(S) && isfield(S, key) && ~isempty(S.(key)) && all(isfinite(S.(key)))
    out = S.(key);
else
    out = def;
end
end
