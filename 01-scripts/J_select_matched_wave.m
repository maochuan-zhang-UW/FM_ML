% Script to filter Felix dataset events, setting W_*, PoML_W_*, CFM_W_*, and EQP_* to empty when they do not match Po_*
% Handles Felix as a non-scalar structure array
% Load the Felix dataset
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/F_ML/A_wave_dB15_DT_CFM_EQP_allAgree.mat')

% Define the stations
stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};

% Initialize counters for kept and removed events
kept_events = zeros(1, length(stations));
removed_events = zeros(1, length(stations));
removed_felix_events = 0;
% Loop through each station
for i = 1:length(stations)
    station = stations{i};
    
    % Field names for ground truth, predictions, and W_*
    po_field = sprintf('Po_%s', station);
    poml_field = sprintf('PoML_W_%s', station);
    cfm_field = sprintf('CFM_W_%s', station);
    eqp_field = sprintf('EQP_%s', station);
    w_field = sprintf('W_%s', station);
    
    % Initialize counters for this station
    station_kept = 0;
    station_removed = 0;
    
    % Loop through each element of the Felix structure array
    for j = 1:length(Felix)
        % Extract data for the current station and element
        po_data = Felix(j).(po_field);
        poml_data = Felix(j).(poml_field);
        cfm_data = Felix(j).(cfm_field);
        eqp_data = Felix(j).(eqp_field);
        w_data = Felix(j).(w_field);

        if isempty(w_data)
            continue;
        end
        
        % Ensure data is in the correct format (row vector)
        if iscolumn(po_data)
            po_data = po_data';
            poml_data = poml_data';
            cfm_data = cfm_data';
            eqp_data = eqp_data';
        end
        
        % Identify indices where all four fields are identical
        matching_idx = (po_data == poml_data) & (po_data == cfm_data) & (po_data == eqp_data);
        
        % Count kept and removed events for this element
        station_kept = station_kept + sum(matching_idx);
        station_removed = station_removed + sum(~matching_idx);
        
        % Set non-matching events to empty
        Felix(j).(poml_field)(~matching_idx) = [];
        Felix(j).(cfm_field)(~matching_idx) = [];
        Felix(j).(eqp_field)(~matching_idx) = [];
        if matching_idx
        Felix(j).(w_field) = [];
        end
    end
    
    % Store total counts for the station
    kept_events(i) = station_kept;
    removed_events(i) = station_removed;
end

% Display results
fprintf('Filtering results for each station:\n');
fprintf('Station\tKept Events\tRemoved Events\n');
for i = 1:length(stations)
    fprintf('%s\t%d\t\t%d\n', stations{i}, kept_events(i), removed_events(i));
end

% Step 2: Remove Felix events where all W_* fields are empty or all NaN
% Create a logical array to mark events for removal
remove_idx = false(1, length(Felix));
for j = 1:length(Felix)
    all_w_empty = true;
    for i = 1:length(stations)
        w_field = sprintf('W_%s', stations{i});
        w_data = Felix(j).(w_field);
        % Check if W_* is empty or all NaN
        if ~isempty(w_data) && any(~isnan(w_data))
            all_w_empty = false;
            break;
        end
    end
    if all_w_empty
        remove_idx(j) = true;
        removed_felix_events = removed_felix_events + 1;
    end
end

% Remove marked Felix events
Felix(remove_idx) = [];

% Display results
fprintf('Filtering results for each station:\n');
fprintf('Station\tKept Events\tRemoved Events\n');
for i = 1:length(stations)
    fprintf('%s\t%d\t\t%d\n', stations{i}, kept_events(i), removed_events(i));
end
fprintf('\nTotal Felix events removed (all W_* empty or NaN): %d\n', removed_felix_events);
fprintf('Remaining Felix events: %d\n', length(Felix));

% Optionally save the modified Felix structure
% save('filtered_Felix_with_removal.mat', 'Felix');
% Optionally save the modified Felix structure
% save('filtered_Felix_empty.mat', 'Felix');

% List of W_* fields to check
wFields = {'W_AS1', 'W_AS2', 'W_CC1', 'W_EC1', 'W_EC2', 'W_EC3', 'W_ID1'};

% Initialize array to store counts for each field
counts = zeros(1, length(wFields));

% Loop through each event in the Felix struct array
for i = 1:length(Felix)
    % Check each W_* field for non-empty data
    for j = 1:length(wFields)
        if ~isempty(Felix(i).(wFields{j}))
            counts(j) = counts(j) + 1;
        end
    end
end

% Display the results
fprintf('Number of non-empty events for each W_* field:\n');
for j = 1:length(wFields)
    fprintf('%s: %d\n', wFields{j}, counts(j));
end