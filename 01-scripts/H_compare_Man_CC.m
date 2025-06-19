clc; clear;
%load('/Users/mczhang/Documents/GitHub/FM3/02-data/E_Po/E_All_polished_updated_V2.mat');

%load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_manual_AS1_CCPo.mat');
path2 = '/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/';

%stations = {'AS1', 'AS2', 'CC1', 'EC1', 'EC2', 'EC3', 'ID1'};
%stations = {'AS1'};\
stations = {'AS1','AS2', 'EC1', 'EC2', 'EC3', 'ID1'};
results = struct();

for s = 1:length(stations)
    station = stations{s};
    out_file = strcat(['/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_manual_', station ,'_CCPo.mat']);
    load(out_file);
    
    po_field = ['Po_' station, ];

    poCC_field = [ station '_CCPo'];

    results.(station).total_comparable = 0;
    results.(station).matching = 0;
    results.(station).ground_truth = [];
    results.(station).predictions = [];
    for i = 1:length(Felix)
        if isempty(Felix(i).(po_field)) || isempty(Felix(i).(poCC_field))
            continue;
        end

        po_value = sign(Felix(i).(po_field));
        poCC_value = sign(Felix(i).(poCC_field));  % use sign just like po_value

        if po_value ~= 0 && poCC_value ~= 0
            results.(station).total_comparable = results.(station).total_comparable + 1;
            results.(station).ground_truth(end+1) = po_value;
            results.(station).predictions(end+1) = poCC_value;

            if po_value == poCC_value
                results.(station).matching = results.(station).matching + 1;
            end
        end


    end
    station = stations{s};  % Only 'AS1' in your case

    compared = results.(station).total_comparable;
    matching = results.(station).matching;
    %agreement = results.(station).agreement;

    fprintf('Compared: %d\n', compared);
    fprintf('Matching: %d\n', matching);
    fprintf('Agreement: %.3f%%\n', matching/compared);

end
