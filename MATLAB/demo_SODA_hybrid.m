clear all
close all
clc
load exampledata
%% Self-organising Direction-Aware Data Partitioning (Hybrid)
%% Use the partitioning result obtained by the offline version as the prime and use the evolving extension to continue the process
%%
% % Seperate the offline training set (Data1) and the online training set (Data2)
Offline_percentage=0.75;
[L,W]=size(data);
L1=round(L*Offline_percentage);
Data1=data(1:1:L1,1:1:W);
L2=L-L1;
Data2=data(L1+1:1:L,1:1:W);
%%
input.GridSize=6; 
input.StaticData=Data1;
% % Input: StaticDat - input data;  GridSize - grid size
% % Use Self-Organising Direction Aware Data Partitioning Algorithm (Offline) as a prime
[output]=SelfOrganisedDirectionAwareDataPartitioning(input,'Offline');
%%
% % Use the Evolving Extension to continue the partitioning process
input.StreamingData=Data2;
input.AllData=data;
input.SystemParams=output.SystemParams;
[output]=SelfOrganisedDirectionAwareDataPartitioning(input,'Evolving');
% % Input: StreamingData       -  input data;  
% %        GridSize            -  grid size; 
% %        SystemParams        -  the offline prime; 
% %        AllData             -  the whole dataset, for creating the final results with the identified prototypes
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
