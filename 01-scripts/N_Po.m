%clc;
clear;close all;
path='/Users/mczhang/Documents/GitHub/FM5_ML/02-data/';
fields={'AS1','AS2','CC1','EC1','EC2','EC3','ID1'};
groups = {'All'};
for gp=1
    for kz=1:length(fields)
        nz=1;
        resultman=[];
        load([path,'M_SVD/M_',fields{kz},'.mat']);
        load([path,'K_aug/',fields{kz},'_add_V4_STEP.mat']);
        eval(strcat('wave = ', fields{kz}, '_add;'));
        eval(strcat('Felix = ', fields{kz}, '_add;'));
        for i = 1:length(wave)
            % Check if wave.Po_fields{kz} is not 0
            if wave(i).(['Man_' fields{kz}]) ~= 0 | ~isempty(wave(i).(['Man_',fields{kz}]))
                % Find the index of wave.ID in SVD_result
                index = find(SVD_result(:,1) == wave(i).ID2);
                % If wave.ID is found in SVD_result, assign wave.Po_fields{kz} to SVD_result(index,6)
                if ~isempty(index)
                    SVD_result(index,7) = wave(i).(['Man_',fields{kz}]);
                end
            end
        end
        % compare with the manual pick and determine the final polarity
        n=1;
        for i = 1:length(wave)
            if wave(i).(['Man_',fields{kz}]) ~= 0 | ~isempty(wave(i).(['Man_',fields{kz}]))
                ind2 = find(SVD_result(:,1) == wave(i).ID2); % this is the row of matrixs
                if isempty(ind2) | abs(wave(i).(['Man_',fields{kz}])) < 1
                    continue;
                end
                wave_test(n).ID=wave(i).ID2;
                wave_test(n).Po=sign(wave(i).(['Man_',fields{kz}]));
                %            wave_test(n).wave=wave(i).(['W_',fields{kz}]);
                wave_test(n).SVD=SVD_result(ind2,2);
                mo(i) = sign(wave(i).(['Man_',fields{kz}])) * SVD_result(ind2,2);
                de(i) = abs(mo(i));
                n=n+1;
            end
        end

        WTcatpol = sum(mo) / sum(de);
        for i=1:length(wave_test)
            wave_test(i).CC_Po=sign(wave_test(i).SVD*WTcatpol);
            wave_test(i).WTcat=WTcatpol;
        end
        ind_un=find([wave_test.CC_Po]==[wave_test.Po]);
        resultman(nz,1)=nz;
        resultman(nz,2)=WTcatpol;
        resultman(nz,3)=length(ind_un)/length(wave_test);
        resultman(nz,4)=length(ind_un);
        resultman(nz,5)=length(wave_test);
        nz=nz+1;


        disp([groups{gp},'_',fields{kz},':',num2str(WTcatpol,3),', Same Ratio:' num2str(length(ind_un)/length(wave_test),2),',N:' ...
            num2str(length(ind_un)),'/',num2str(length(wave_test)),', Value: ' num2str(sum(de),2), ', out of: ' num2str(sum(abs(SVD_result(:,2))),2) ...
            ', percent: ' num2str(sum(de)/sum(abs(SVD_result(:,2)))*100,4), ',CC matched:' num2str(sum(SVD_result(:,4))/(sum(SVD_result(:,4))+sum(SVD_result(:,5))),2)]);
        %%
        % % Define the file path
        % filePath = [path,'D_man/D_',groups{gp},'_',fields{kz},'_uncon.mat'];
        % % Check if the file exists to avoid errors
        % if exist(filePath, 'file') == 2
        %     % Delete the file
        %     delete(filePath);
        %     %disp(['File ', filePath, ' deleted successfully.']);
        % else
        %     disp(['File ', filePath, ' does not exist.']);
        % end
        %%
        %save([path,'D_man/D_',groups{gp},'_',fields{kz},'_uncon.mat'], 'wave_test');clear wave_test;
        SVD_result(:,8) = sign(SVD_result(:,3) * sign(WTcatpol));    % Loop over each element in the wave structure
        % remove less than threshold
        SVD_result(find(abs(SVD_result(:,2))<10^(SVD_result(1,6)+2)),8)=0;
        % compare with the manual pick and determine the final polarity

        for j = 1:length(SVD_result)
            ind3 = find([Felix.ID2]==SVD_result(j,1)); % this is the row of matrixs
            if ~isempty(ind3)
                Felix(ind3).(['CC_', fields{kz}])=SVD_result(j,8) ;

            end
        end
        clear mo de SVD_result wave wave_test;
        save([path,'N_Po/N_Po' fields{kz} 'V4c.mat'], 'Felix');
    end
    
end



