clc;clear all;
yalmip clear;
options = sdpsettings('verbose', 1, 'dualize', 0);
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



%% Localized line failures (8 lines)
% 27-28 (#34), 28-29 (#35), 29-31 (#40), 31-32 (#42), 27-32 (#43), 27-115 (#181),
% 32-114 (#180), 114-115 (#182)

index = [34,35,40,42,43,180,181,182];
N = 7;
zero_NN = dec2bin(0:2^N-1)' - '0'; % 128 scenarios, line #43 is assumed to be already open due to wildfire
zero_N =[zero_NN(1:4,:);zeros(1,128);zero_NN(5:7,:)];


gamma = 100;
k = 1; % scenario number selected
rr = randperm(128,k); % randomly select k scenarios





%% Generate the xi vector for these scenarios
xi{1} = ones(186,1);




%% Decision Variables
for i = 1:k
    theta{i} = sdpvar(118,1); % decision variable
    P_nk{i} = sdpvar(186,1); % decision variable
    P_s{i} = sdpvar(118,1); % load shedding
end

P_ng = sdpvar(118,1); % decision variable

z = binvar(186,1); % decision variable

P_nd = mpc.bus(:,3); % load
P_nd(90) = 440;
%P_nd = P_nd*1.4;

B = 1./mpc.branch(:,4); % DC approximation: 1/X 
B1 = [];
for i=1:186
    B1(i) = mpc.branch(i,4)/((mpc.branch(i,4))^2 + (mpc.branch(i,3))^2);
end


%% objective function
obj = 0;
for i = 1:k
    obj = obj + gamma*sum(P_s{i});
end
obj = obj/k;
obj = obj + c'*P_ng;




%% constraint 1 (phase angle)
constraints = {};
theta_min = -ones(118,1);
theta_max = ones(118,1);

for i = 1:k
    constraints{end+1} = theta_min <= theta{i} <= theta_max;
    constraints{end+1} = theta{i}(1) == 0;
end


%% constraint 2 (generation)
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
constraints{end+1} = P_ng_min <= P_ng <= P_ng_max;




%% constraint 3
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
for i = 1:k
    constraints{end+1} = xi{i}.*(-P_nk_max).*z <= P_nk{i} <= xi{i}.*(P_nk_max).*z;
end



%% constraint 4 (power balance)
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

for i = 1:k
    constraints{end+1} = A*P_nk{i} == P_ng  - (P_nd - P_s{i});
end



for i = 1:k
    for j = 1:118
        constraints{end+1} = P_s{i}(j) >= 0;
    end
end




%% constraint 5 (Big M constraints)
M = zeros(186,118);
for i=1:186
    tmp1 = mpc.branch(i,1);
    tmp2 = mpc.branch(i,2);
    M(i,tmp1) = B(i);
    M(i,tmp2) = -B(i);
end


for i = 1:k
    for j = 1:186
        constraints{end+1} = M(j,:)*theta{i}*100 - P_nk{i}(j) + (1-z(j)*xi{i}(j))*1000 >= 0;
        constraints{end+1} = M(j,:)*theta{i}*100 - P_nk{i}(j) - (1-z(j)*xi{i}(j))*1000 <= 0;
    end
end

%% constraint 6 (number of switching lines)
constraints{end+1} = ones(1,186)*(ones(186,1) - z) <= 3;

 
 
%% solve the problem
optimize([constraints{:}], obj, options);
obj_exact = value(obj)
P_ng = value(P_ng);


z = value(z)



obj = value(obj_exact)




