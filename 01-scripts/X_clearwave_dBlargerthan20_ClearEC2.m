
clc;clear;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_dB15.mat');
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
numStations = length(stations);
noise = [];
 n=1;
 noiseThreshold=15;
 % for i = 1:length(Felix)
 %     for j = 5
 %         nspField = ['NSP_' stations{j}];
 %         wField   = ['W_' stations{j}];
 %         if ~isempty(Felix(i).(nspField)) && isfield(Felix(i), wField)
 %             nsp = Felix(i).(wField);
 %             noi=mean(abs(nsp(1:80)));
 %             noise(n)=noi;
 %             n=n+1;
 %             if noi > noiseThreshold
 %                 Felix(i).(wField)   = [];   % clear waveform
 %                 %                    Felix(i).(nspField) = [];   % optional: also clear NSP
 %             end
 % 
 %         end
 % 
 %     end
 % end

figure;
for j = 1:numStations
    subplot(2,4,j)
    for i=1:length(Felix)
        wField = ['W_' stations{j}];
        if isfield(Felix(i), wField) && ~isempty(Felix(i).(wField))
            plot(normalize(Felix(i).(wField)));
            hold on;
        end
    end
end