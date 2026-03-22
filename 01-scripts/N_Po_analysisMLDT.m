% Load all 7 station files (they follow the N_Po*_CFM.mat pattern)
clear
stations = {'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
results = cell(numel(stations),4); % {Station, TotalValid, Same, Ratio}

for i = 1:numel(stations)
    st = stations{i};
    fname = ['/Users/mczhang/Documents/GitHub/FM5_ML/02-data/N_Po/N_Po' st '_DT.mat'];
    load(fname,'Felix');
    %Felix = [Felix{:}];  % ensure array
    
    % Build field names
    cfm_field = ['PoML_W_' st];
    man_field = ['Man_' st];
    
    % Extract values
    cfm = [Felix.(cfm_field)]';
    man = [Felix.(man_field)]';
    
    % Mask for valid CFM ≠ 0
    mask = cfm ~= 0;
    cfm_valid = cfm(mask);
    man_valid = man(mask);
    
    % Count same / different
    num_same = sum(cfm_valid == man_valid);
    num_total = numel(cfm_valid);
    ratio = num_same / num_total;
    
    % Save results
    results{i,1} = st;
    results{i,2} = num_total;
    results{i,3} = num_same;
    results{i,4} = ratio;
end

% Convert to table for pretty display
T = cell2table(results, 'VariableNames', {'Station','TotalValid','Same','Ratio'});
disp(T);
