% ---------------------------------------------------------
% Resample W_* fields from 200 Hz → 100 Hz
% Window: [-0.32, 0.32] s with 64 points
% Edge fill: [-0.32, -0.26] ← [-0.25, -0.19]
clc;clear;
% ---------------------------------------------------------
load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated.mat')
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};

fs_old = 200;              % original sampling rate (Hz)
fs_new = 100;              % target sampling rate (Hz)

t_old = -0.25 : 1/fs_old : 1.0 - 1/fs_old;   % 250 samples
t_new = linspace(-0.32, 0.32, 64);            % exactly 64 points

for i = 1:numel(Felix)
    for j = 1:numel(stations)

        fname = ['W_' stations{j}];

        % Skip if field does not exist or is empty
        if ~isfield(Felix, fname) || isempty(Felix(i).(fname))
            continue
        end

        w = Felix(i).(fname)(:);  % ensure column vector

        % Safety check
        if numel(w) ~= numel(t_old)
            warning('Event %d %s length mismatch, skipped', i, fname);
            continue
        end

        % ---- Interpolate original waveform ----
        % Use linear interpolation, extrapolate disabled
        w_interp = interp1(t_old, w, t_new, 'linear', 'extrap');

        % ---- Edge fix: [-0.32, -0.26] ← [-0.25, -0.19] ----
        idx_target = (t_new >= -0.32 & t_new < -0.26);
        t_source   = t_new(idx_target) + 0.07;   % shift to [-0.25, -0.19]

        w_interp(idx_target) = interp1(t_old, w, t_source, 'linear');

        % ---- Save back to original field ----
        Felix(i).(fname) = w_interp(:);
    end
end
