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



%% Number of topology scenarios
% k = 4;
% 
% xi{1} = ones(186,1);
% xi{1}(33) = 0;
% xi{1}(34) = 0;
% xi{1}(35) = 0;
% xi{1}(36) = 0;
% 
% 
% xi{2} = ones(186,1);
% xi{2}(23) = 0;
% xi{2}(24) = 0;
% xi{2}(25) = 0;
% xi{2}(26) = 0;
% 
% 
% xi{3} = ones(186,1);
% xi{3}(31) = 0;
% xi{3}(33) = 0;
% xi{3}(37) = 0;
% xi{3}(39) = 0;
% 
% 
% xi{4} = ones(186,1);
% xi{4}(13) = 0;
% xi{4}(24) = 0;
% xi{4}(55) = 0;
% xi{4}(56) = 0;

% xi matrix: for each row, there are 4 numbers, corresponding to the
% opening lines due to wildfire (30 scenarios)
% xi_matrix =[161,108,148,67;183,47,2,125;145,130,60,3;90,55,37,120;142,126,36,119;33,135,85,20;159,139,109,171;103,94,2,52;35,48,150,60;41,91,55,36;168,105,142,92;44,109,87,15;83,120,130,156;153,175,92,151;8,106,26,5;129,6,92,178;155,111,24,145;111,84,125,81;149,163,157,80;81,69,173,5;155,76,65,117;17,120,82,135;77,84,147,78;109,32,26,174;185,145,111,99;135,153,49,110;125,141,130,108;58,164,85,30;125,91,14,165;45,89,72,71];
% 
% % % xi matrix: for each row, there are 4 numbers, corresponding to the
% % % opening lines due to wildfire (100 scenarios)
% % xi_matrix = [97,99,65,141;34,54,17,88;19,180,48,143;28,137,36,47;50,118,31,2;61,131,39,186;42,151,170,56;52,73,138,183;112,119,22,2;77,27,144,106;12,123,109,27;94,114,64,98;110,165,155,94;95,84,158,19;66,136,85,6;168,112,44,143;29,42,135,1;90,51,26,84;48,123,88,31;136,65,92,54;76,113,146,86;16,177,72,2;42,172,91,76;36,159,50,72;10,9,32,91;69,131,72,81;27,20,28,118;62,102,108,161;182,60,39,140;101,120,12,69;71,117,181,133;157,173,34,77;151,37,78,10;184,17,143,79;64,41,1,166;32,105,127,57;22,180,85,32;93,173,172,44;47,32,91,100;147,148,78,9;141,20,177,80;118,93,146,17;56,87,154,18;112,119,59,178;113,8,6,154;130,48,161,94;23,67,8,20;153,144,128,38;160,151,78,117;168,184,5,30;28,184,43,140;22,36,61,97;24,3,174,183;120,39,23,35;61,152,51,31;92,43,173,103;43,143,7,101;139,182,58,183;1,63,107,57;108,30,104,70;40,2,141,15;56,172,81,27;184,91,66,131;116,39,127,85;104,54,50,129;114,14,63,176;57,164,64,131;41,175,15,38;76,135,181,123;36,115,45,25;58,9,98,116;80,7,96,93;60,134,145,43;39,120,106,131;186,97,176,32;56,35,32,112;54,144,100,38;29,22,185,162;182,95,18,144;47,52,83,141;153,22,54,166;54,111,118,126;167,116,154,152;126,132,174,162;121,32,174,92;60,69,93,80;178,97,177,148;88,3,92,145;105,123,55,59;7,114,130,136;158,177,140,110;57,96,20,149;45,120,27,76;116,25,110,126;15,55,3,136;136,184,138,39;105,144,45,37;143,85,155,165;184,119,14,185;122,174,42,94];
% 
% k = 30;
% for i = 1:k
%     xi{i} = ones(186,1);
%     xi{i}(xi_matrix(i,:)) = 0;
% end



%% Localized line failures (8 lines)
% 27-28 (#34), 28-29 (#35), 29-31 (#40), 31-32 (#42), 27-32 (#43), 27-115 (#181),
% 32-114 (#180), 114-115 (#182)

index = [34,35,40,42,43,180,181,182];
N = 7;
zero_NN = dec2bin(0:2^N-1)' - '0'; % 128 scenarios, line #43 is assumed to be already open due to wildfire
zero_N =[zero_NN(1:4,:);zeros(1,128);zero_NN(5:7,:)];


gamma = 100;
k = 5; % scenario number selected
rr = randperm(128,k); % randomly select k scenarios





%% Generate the xi vector for these scenarios
for i = 1:k
    xi{i} = ones(186,1);
    for j = 1:N
        xi{i}(index(j)) = zero_N(j,rr(i));
    end
end




%% Decision Variables
for i = 1:k
    theta{i} = sdpvar(118,1); % decision variable
    P_nk{i} = sdpvar(186,1); % decision variable
    P_s{i} = sdpvar(118,1); % load shedding
    delta_g{i} = sdpvar(118,1); % slack adjustment for generation
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
    obj = obj + gamma*sum(P_s{i}) + (ones(118,1)'*abs(delta_g{i}));
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
%constraints{end+1} = P_ng_min <= P_ng <= P_ng_max;

for i = 1:k
    constraints{end+1} = P_ng_min <= P_ng + delta_g{i}  <= P_ng_max;
end


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
    constraints{end+1} = A*P_nk{i} == P_ng + delta_g{i} - (P_nd - P_s{i});
end

for i = 1:k
    for j = 1:118
        constraints{end+1} = -5 <= delta_g{i}(j) <= 5;
    end
end



% for i = 1:k
%     constraints{end+1} = A*P_nk{i} == P_ng - (P_nd - P_s{i});
% end

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
%sum(P_s{1})
P_s{1} = value(P_s{1})
P_s{2} = value(P_s{2})
P_s{3} = value(P_s{3})
P_s{4} = value(P_s{4})


%P_S = [P_s{1} P_s{2} P_s{3} P_s{4} P_s{5} P_s{6} P_s{7} P_s{8}];


z = value(z)
% sum(P_S)
% fig = gcf;
% fig.Color = [1 1 1];
% p = plot(P_S)
% 
% xlabel('Bus index','FontSize',16,'FontWeight','bold') 
% ylabel('Amount of load shedding (MW)','FontSize',16,'FontWeight','bold') 
% set(gca,'FontSize',16)
% 
% % p.Marker = 'x';
% % p.MarkerSize = 14;
% p.LineWidth = 1.5;


obj = value(obj_exact)



% for i = 1:k
%     P_s{i} = value(P_s{i});
%     plot(P_s{i})
%     hold on
% end


delta = value(delta_g{1})


obj1 = c'*P_ng;
obj1 = value(obj1)

obj2 = 0;
for i = 1:k
    obj2 = obj2 + sum(P_s{i});
end
obj2 = value(obj2)/k

obj3 = 0;
for i = 1:k
    obj3 = obj3 + sum(c'*abs(delta_g{i}));
end
obj3 = value(obj3)/k


AAA = [];
BBB = [];
for i = 1:k
    tmp1 = sum(value(delta_g{i}));
    AAA = [AAA tmp1];
    tmp2 = sum(value(P_s{i}));
    BBB = [BBB tmp2];
end


plot(AAA);
hold on;
plot(BBB);

obj1 = value(obj1)
loadshed_mean = mean(BBB)



