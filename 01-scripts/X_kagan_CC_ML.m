clc; clear; close all;
lonLim = [-130.03 -129.97];
latLim = [45.92 45.97];
%% ===========================================
% PART 1 — LOAD 1D/3D KAGAN DATA
%% ===========================================
load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_HASH_All_ML_sameClusterasbeforev_confidence.mat');
%load('/Users/mczhang/Documents/GitHub/FM4/02-data/Before22OBSs/G_FM/G_HASH_All_ML_sameClusterasbefore.mat');
%% ---- Compute REAL Kagan angles (1-D vs 3-D) ----
event = event1;
event([event.mechqual] == 'D' | [event.mechqual] == 'C') = [];
clear event1;
load('/Users/mczhang/Documents/GitHub/FM3/02-data/G_FM/G_2015Erp_polished.mat');
%load('/Users/mczhang/Documents/GitHub/FM4/02-data/Before22OBSs/G_FM/G_HASH_All.mat');
%event1 = event1;clear event1;
for i = 1:length(event)
    ind = find([event1.id] == event(i).id );
    if ~isempty(ind)% && ind(1) <= 4090 
        event(i).kg = kagan( ...
            [event(i).avmech(1), event(i).avmech(2), event(i).avmech(3)], ...
            [event1(ind(1)).avmech(1), event1(ind(1)).avmech(2), event1(ind(1)).avmech(3)]);
    end
end

event(arrayfun(@(x) isempty(x.kg), event)) = [];
data_real = real([event.kg]);
figure;
hist(data_real);

