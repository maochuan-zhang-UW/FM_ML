% export_fig08_published_mechanisms.m
%
% Writes 03-figs/fig08_published_mechanisms.csv, the table that
% 01-scripts/figures/fig08_mechanisms.m draws Fig. 8 from: the composite focal
% mechanisms of the clusters resolved from both cross-correlation (CC) and
% AxialPolCap polarities, with the Kagan angle between the two.
%
% This script documents how the table was made and cannot be run from the data
% archive. Its inputs are the two HASH mechanism catalogues
%   02-data/G_HASH_All_ML_sameClusterasbeforev_confidence.mat   (AxialPolCap)
%   <FM4>/02-data/FMs_Composite_Single_SK_HASH_Final.mat        (CC)
% and kagan.m (P. Kolar's translation of Kagan 1991), which is not redistributed.
%
%   matlab -batch "run('01-scripts/figures/provenance/export_fig08_published_mechanisms.m')"

clear; clc;
repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
fm4Root  = '/Users/mcZhang/Documents/GitHub/FM4';
addpath(fullfile(fm4Root, '01-scripts'));                       % kagan.m

M = load(fullfile(repoRoot, '02-data', 'G_HASH_All_ML_sameClusterasbeforev_confidence.mat'), 'event1');
C = load(fullfile(fm4Root,  '02-data', 'FMs_Composite_Single_SK_HASH_Final.mat'), 'eventHASH_org');
ml = M.event1;  cc = C.eventHASH_org;
[ids, iML, iCC] = intersect(arrayfun(@(e) double(e.id), ml), arrayfun(@(e) double(e.id), cc));
ml = ml(iML);  cc = cc(iCC);  n = numel(ids);

vec3 = @(v) reshape([double(v(:)); nan(3,1)], 1, []);           % pad empty fields with NaN
T = table();
T.cluster = ids(:);
T.time    = string(datestr(arrayfun(@(e) double(e.time), ml(:)), 'yyyy-mm-ddTHH:MM:SS'));
T.lat     = arrayfun(@(e) double(e.lat), ml(:));
T.lon     = arrayfun(@(e) double(e.lon), ml(:));
src = {cc, ml};  tag = {'cc', 'ml'};
for s = 1:2
    ev = src{s};
    sdr = nan(n,3); fn = nan(n,3); sl = nan(n,3); q = strings(n,1); ft = strings(n,1);
    for i = 1:n
        v = vec3(ev(i).avmech);   sdr(i,:) = v(1:3);
        v = vec3(ev(i).avfnorm);  fn(i,:)  = v(1:3);
        v = vec3(ev(i).avslip);   sl(i,:)  = v(1:3);
        q(i)  = string(char(ev(i).mechqual(1)));
        ft(i) = string(strtrim(char(ev(i).faultType)));
    end
    T.(['strike_' tag{s}]) = sdr(:,1);  T.(['dip_' tag{s}]) = sdr(:,2);  T.(['rake_' tag{s}]) = sdr(:,3);
    T.(['quality_' tag{s}]) = q;        T.(['fault_type_' tag{s}]) = ft;
    T.(['fnorm_e_' tag{s}]) = fn(:,1);  T.(['fnorm_n_' tag{s}]) = fn(:,2);  T.(['fnorm_u_' tag{s}]) = fn(:,3);
    T.(['slip_e_' tag{s}])  = sl(:,1);  T.(['slip_n_' tag{s}])  = sl(:,2);  T.(['slip_u_' tag{s}])  = sl(:,3);
end
kg = nan(n,1);
for i = 1:n
    a = [T.strike_ml(i) T.dip_ml(i) T.rake_ml(i)];  b = [T.strike_cc(i) T.dip_cc(i) T.rake_cc(i)];
    if all(isfinite([a b])), kg(i) = kagan(a, b); end
end
T.kagan = kg;

out = fullfile(repoRoot, '03-figs', 'fig08_published_mechanisms.csv');
writetable(T, out);
fprintf('wrote %s (%d paired clusters; median Kagan angle %.1f deg)\n', out, n, median(kg, 'omitnan'));
