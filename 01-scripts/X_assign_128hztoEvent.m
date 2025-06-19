clc;clear;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/D_man/D_CC1.mat');
Felix1=Felix;clear Felix;
load('/Users/mczhang/Documents/GitHub/FM5_ML/02-data/A_wave/A_wave_forML128.mat')
[I,IA,IB]=intersect([Felix.ID],[Felix1.ID]);
for i=1:length(IB)
    Felix1(IB(i)).W1_CC1=Felix(IA(i)).W_CC1;
end


