clc;clear all;
mpc = loadcase('case_RTS_GMLC');

T = readtable('bus.csv');
TT = T(:,[1,14,15]);
G = table2array(TT);
branch = mpc.branch;

for i = 1:120
    from = branch(i,1);
    to = branch(i,2);
    latSeattle = G(find(G(:,1) == from),2);
    lonSeattle = G(find(G(:,1) == from),3);
    latAnchorage = G(find(G(:,1) == to),2);
    lonAnchorage = G(find(G(:,1) == to),3);
    geoplot([latSeattle latAnchorage],[lonSeattle lonAnchorage],'b-*');
    geolimits([32 35],[-120 -110])
    geobasemap streets
    hold on;
end

% i = 20
%     from = branch(i,1);
%     to = branch(i,2);
%     latSeattle = G(find(G(:,1) == from),2);
%     lonSeattle = G(find(G(:,1) == from),3);
%     latAnchorage = G(find(G(:,1) == to),2);
%     lonAnchorage = G(find(G(:,1) == to),3);
%     geoplot([latSeattle latAnchorage],[lonSeattle lonAnchorage],'r-^');
%     
% hold on;
%     
% i = 22
%     from = branch(i,1);
%     to = branch(i,2);
%     latSeattle = G(find(G(:,1) == from),2);
%     lonSeattle = G(find(G(:,1) == from),3);
%     latAnchorage = G(find(G(:,1) == to),2);
%     lonAnchorage = G(find(G(:,1) == to),3);
%     geoplot([latSeattle latAnchorage],[lonSeattle lonAnchorage],'r-^');
%     
% hold on;
%     
% i = 92
%     from = branch(i,1);
%     to = branch(i,2);
%     latSeattle = G(find(G(:,1) == from),2);
%     lonSeattle = G(find(G(:,1) == from),3);
%     latAnchorage = G(find(G(:,1) == to),2);
%     lonAnchorage = G(find(G(:,1) == to),3);
%     geoplot([latSeattle latAnchorage],[lonSeattle lonAnchorage],'r-^');
%     
% hold on;
%     
% i = 93
%     from = branch(i,1);
%     to = branch(i,2);
%     latSeattle = G(find(G(:,1) == from),2);
%     lonSeattle = G(find(G(:,1) == from),3);
%     latAnchorage = G(find(G(:,1) == to),2);
%     lonAnchorage = G(find(G(:,1) == to),3);
%     geoplot([latSeattle latAnchorage],[lonSeattle lonAnchorage],'r-^');

hold off;

geolimits([31 37],[-120 -112])

writetable(T, 'RTS_bus_data.csv');

