clc
close all
load exampledata
%% Self-organising Direction-Aware Data Partitioning (offline version)
input.GridSize=6; % % The gird size, which decide the level of granularity of the partitioning results. The larger the girdsize is, the more detailed partitioning result the algoirthm will obtain
input.StaticData=data;
[output]=SelfOrganisedDirectionAwareDataPartitioning(input,'Offline');
% % Self-Organising Direction Aware Data Partitioning Algorithm (Offline)
% % Input: StaticDat - input data;  GridSize - grid size
% % Output:   C            -   Identified prototypes 
% %           IDX          -   Data cloud label (Clustering label, classes) of the data samples
% %           SystemParams -   Meta-parameters of the identified direction-aware planes, important for the subsequent processing with the evolving extension
%% Plot the partitioning result
figure
T=unique(output.IDX);
for i=1:1:length(T)
plot(data(output.IDX==T(i),1),data(output.IDX==T(i),2),'.','linewidth',2,'markersize',15)
hold on
end
hold on
grid on
xlabel('x-axis')
ylabel('y-axis')
set(gca, 'FontSize', 14)
plot(output.C(:,1),output.C(:,2),'k*','linewidth',2,'markersize',8)
hold off
