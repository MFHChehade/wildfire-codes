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
    geoplot([latSeattle latAnchorage],[lonSeattle lonAnchorage],'k-.');
    geolimits([32 35],[-120 -110])
    geobasemap streets
    hold on;
end

%% Wildfire One
x = 34.31267;
y = -117.34635;
lat = zeros(10000,1);
lon = zeros(10000,1);
for i = 1:10000
    lat(i) = x + 0.06*(normrnd(0,2));
    lon(i) = y + 0.06*(normrnd(0,2));
end
weights = 10.*rand(10000,1)
    
geodensityplot(lat,lon,weights,'FaceColor','r')
hold on;

%% Wildfire Two
x = 33.59823;
y = -116.23339;
lat = zeros(10000,1);
lon = zeros(10000,1);
for i = 1:10000
    lat(i) = x + 0.06*(normrnd(0,2));
    lon(i) = y + 0.06*(normrnd(0,2));
end
weights = 10.*rand(10000,1)
    
geodensityplot(lat,lon,weights,'FaceColor','r')
hold on;


%% Wildfire Three
x = 34.010838;
y = -114.484173;
lat = zeros(10000,1);
lon = zeros(10000,1);
for i = 1:10000
    lat(i) = x + 0.06*(normrnd(0,2));
    lon(i) = y + 0.06*(normrnd(0,2));
end
weights = 10.*rand(10000,1)
    
geodensityplot(lat,lon,weights,'FaceColor','r')
hold on;




hold off;

geolimits([31 37],[-120 -112])




