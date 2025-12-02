clc;clear all;
yalmip clear;
%options = sdpsettings('verbose', 1, 'dualize', 0, 'solver', 'gurobi','gurobi.MIPGap',0.01);
options = sdpsettings('verbose', 1, 'dualize', 0, 'solver', 'gurobi', 'gurobi.InfUnbdInfo',1,'savesolveroutput ',1);
mpc = rundcpf('case118');

%% Generation Costs
c = zeros(118,1);
c(10) = 0.217;
c(12) = 1.052;
c(25) = 0.434;
c(26) = 0.308;
c(31) = 5.882;
c(46) = 3.448;
c(49) = 0.467;
c(54) = 1.724;
c(59) = 0.606;
c(61) = 0.588;
c(65) = 0.2493;
c(66) = 0.2487;
c(69) = 0.1897;
c(80) = 0.205;
c(87) = 7.142;
c(92) = 10;
c(100) = 0.381;
c(103) = 2;
c(111) = 2.173;


index = [34,35,40,42,43,180,181,182];
N = 7;
zero_NN = dec2bin(0:2^N-1)' - '0'; % 128 scenarios, line #43 is assumed to be already open due to wildfire
zero_N =[zero_NN(1:4,:);zeros(1,128);zero_NN(5:7,:)];


gamma = 2;
k = 4; % scenario number selected
rr = randperm(128,k); % randomly select k scenarios

%% Generate the xi vector for these scenarios
for i = 1:k
    xi{i} = ones(186,1);
    for j = 1:N
        xi{i}(index(j)) = zero_N(j,rr(i));
    end
end

%% Second Stage Decision Variables
for i = 1:k
    theta{i} = sdpvar(118,1); % decision variable
    P_nk{i} = sdpvar(186,1); % decision variable
    P_s{i} = sdpvar(118,1); % load shedding
    psi{i} = sdpvar(1,1); % variables in benders decomposition
end

P_nd = mpc.bus(:,3); % load
P_nd(90) = 440;


B = 1./mpc.branch(:,4); % DC approximation: 1/X 
B1 = [];
for i=1:186
    B1(i) = mpc.branch(i,4)/(    (mpc.branch(i,4))^2 + (mpc.branch(i,3))^2   );
end


%% Constraints
constraints = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization of x and psi
zr = ones(186,1);
P_ngr = zeros(118,1);

% for i = 1:k
%     psir{i} = 0;
% end


% zr = [1;1;1;1;1;1;1;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;0;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1;1];
% P_ngr = 0.9.*[0;0;0;0;0;0;0;0;0;550;0;0;0;0;0;0;0;0;0;0;0;0;0;0;320;414;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;304;0;0;0;0;0;0;0;0;0;0;0;213.800000000000;0;0;0;491;492;0;0;805.200000000000;0;0;0;0;0;0;0;0;0;0;577;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;352;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


theta_min = -ones(118,1);
theta_max = ones(118,1);

P_nk_max = 220*ones(186,1);
P_nk_max(3) = 440;
P_nk_max(7) = 1100;
P_nk_max(8) = 880;
P_nk_max(9) = 1100;
P_nk_max(21) = 440;
P_nk_max(31) = 440;
P_nk_max(33) = 440;
P_nk_max(38) = 660;
P_nk_max(36) = 660;
P_nk_max(50) = 440;
P_nk_max(51) = 660;
P_nk_max(96) = 440;
P_nk_max(98) = 440;
P_nk_max(99) = 440;
P_nk_max(90) = 440;
P_nk_max(93) = 440;
P_nk_max(94) = 440;
P_nk_max(97) = 440;
P_nk_max(107) = 440;
P_nk_max(183) = 440;
P_nk_max(108) = 440;
P_nk_max(116) = 440;
P_nk_max(123) = 440;
P_nk_max(137) = 440;
P_nk_max(138) = 660;
P_nk_max(140) = 660;
P_nk_max(163) = 440;

P_ng_min = zeros(118,1);
P_ng_max = zeros(118,1);
P_ng_max(10) = 550;
P_ng_max(12) = 185;
P_ng_max(25) = 320;
P_ng_max(26) = 414;
P_ng_max(31) = 107;
P_ng_max(46) = 119;
P_ng_max(49) = 304;
P_ng_max(54) = 148;
P_ng_max(59) = 255;
P_ng_max(61) = 260;
P_ng_max(65) = 491;
P_ng_max(66) = 492;
P_ng_max(69) = 805.2;
P_ng_max(80) = 577;
P_ng_max(87) = 104;
P_ng_max(92) = 100;
P_ng_max(100) = 352;
P_ng_max(103) = 140;
P_ng_max(111) = 136;


A = zeros(118,186); % node power flow net injection matrix
for i=1:118
    for j=1:186
        if(mpc.branch(j,1) == i)
            A(i,j) = 1;
        elseif(mpc.branch(j,2) == i)
            A(i,j) = -1;
        end
    end
end


M = zeros(186,118);
for i=1:186
    tmp1 = mpc.branch(i,1);
    tmp2 = mpc.branch(i,2);
    M(i,tmp1) = B(i);
    M(i,tmp2) = -B(i);
end


result = [];
obj_sub = [];


%% Basic constraints for master problem
P_ng_benchmark = [0;0;0;0;0;0;0;0;0;550;0;0;0;0;0;0;0;0;0;0;0;0;0;0;283.356780710232;414;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;303.967760585794;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;491;492;0;0;805.200000000000;0;0;0;0;0;0;0;0;0;0;577;0;0;0;0;0;0;0;0;0;0;0;1.13686837721616e-13;0;0;0;0;0;0;0;352;0;0;140;0;0;0;0;0;0;0;69.4754587039734;0;0;0;0;0;0;0];


P_ng = sdpvar(118,1); % decision variable
z = binvar(186,1); % decision variable
con= {};
con{end+1} = ones(1,186)*(ones(186,1) - z) <= 2;
con{end+1} = P_ng_min <= P_ng <= P_ng_max;
con{end+1} = sum(P_ng) <= sum(P_nd);
%con{end+1} = norm(P_ng - P_ng_benchmark,inf) <= 20;

obj_master = c'*P_ng;
for i = 1:k
    obj_master = obj_master + gamma*(1/k)*psi{i};
end


iter = 1;

z_history = zeros(186,iter);

for kk = 1:iter
    %% solve subproblem
    for i = 1:k
        constraints = {};
        dual_value{i} = [];
        constraints{end+1} = theta{i} >=theta_min;
        constraints{end+1} = theta{i} <= theta_max;
        constraints{end+1} = theta{i}(1) == 0;
        constraints{end+1} = P_nk{i} >=xi{i}.*(-P_nk_max).*zr;
        constraints{end+1} = P_nk{i} <= xi{i}.*(P_nk_max).*zr;
        for j = 1:186
           constraints{end+1} = M(j,:)*theta{i}*100 - P_nk{i}(j) + (1-zr(j)*xi{i}(j))*1000 >= 0;
           constraints{end+1} = M(j,:)*theta{i}*100 - P_nk{i}(j) - (1-zr(j)*xi{i}(j))*1000 <= 0;
        end
        constraints{end+1} = A*P_nk{i} - P_s{i} == P_ngr - (P_nd);
        for j = 1:118
           constraints{end+1} = P_s{i}(j) >= 0;
        end
        %obj = gamma*ones(118,1)'*P_s{i} + 1000*ones(118,1)'*(abs(P_s{i}) - P_s{i}); % allow negative load shedding, but with high penalty
        obj = gamma*ones(118,1)'*P_s{i}
            
        result_sub = optimize([constraints{:}], obj, options);
        obj_sub = [obj_sub value(obj)];
        for s = 1:378
            dual_value{i} = [dual_value{i};dual(constraints{s})]; % get the optimal dual values for each scenario
        end
        dual_1{i} = dual_value{i}(1:118);
        dual_2{i} = dual_value{i}(119:236);
        dual_3{i} = dual_value{i}(237);
        dual_4{i} = dual_value{i}(238:423);
        dual_5{i} = dual_value{i}(424:609);
        dual_6{i} = dual_value{i}(610:795);
        dual_7{i} = dual_value{i}(796:981);
        dual_8{i} = dual_value{i}(982:1099);
        %dual_9{i} = dual_value{i}(1100:1217);
    %% solve master problem
    con{end+1} = -(theta_min'*dual_1{i} + (theta_max)'*dual_2{i} + (-P_nk_max.*xi{i}.*z)'*dual_4{i} + (P_nk_max.*xi{i}.*z)'*dual_5{i} + (-(ones(186,1) - xi{i}.*z)*1000)'*dual_6{i} + ((ones(186,1) - xi{i}.*z)*1000)'*dual_7{i} + (P_ng - P_nd)'*dual_8{i}) <= psi{i};
    rr = value(-(theta_min'*dual_1{i} + (theta_max)'*dual_2{i} + (-P_nk_max.*xi{i}.*zr)'*dual_4{i} + (P_nk_max.*xi{i}.*zr)'*dual_5{i} + (-(ones(186,1) - xi{i}.*zr)*1000)'*dual_6{i} + ((ones(186,1) - xi{i}.*zr)*1000)'*dual_7{i} + (P_ngr - P_nd)'*dual_8{i}))
    end
    optimize([con{:}], obj_master, options);
    zzz{kk} = value(z);
    zr = value(z);
   %P_ngr = (0.3 + kk*0.1).*value(P_ng);
    P_ngr = value(P_ng);
    z_history(:,kk) = zr;
end

P_ng = value(P_ng);

% obj_sub(end) = [];
% obj_sub = c'*P_ngr + mean(obj_sub);






