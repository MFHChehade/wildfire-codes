clc;clear all;
% tsunamis = readtable('tsunamis.xlsx');
% lat = tsunamis.Latitude;
% lon = tsunamis.Longitude;
% weights = tsunamis.MaxHeight;
% 
% geodensityplot(lat,lon,weights)
% geolimits([41.2 61.4],[-148.6 -107.0])
% geobasemap topographic


x = 34.31267;
y = -117.34635;
lat = zeros(100,1);
lon = zeros(100,1);
for i = 1:100
    lat(i) = x + 0.2*(rand - 0.5);
    lon(i) = y + 0.2*(rand - 0.5);
end
weights = 10.*rand(100,1)
    
geodensityplot(lat,lon,weights)
geobasemap topographic

