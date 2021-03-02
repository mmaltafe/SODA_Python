%% Copyright (c) 2018, Plamen P. Angelov and Xiaowei Gu
%% All rights reserved. Please read the "license.txt" for license terms.
%% This code is the Self-Organised Direction Aware Data Partitioning Algorithm (SODA) described in:
%==========================================================================================================
% X. Gu, P. Angelov, D. Kangin, J. Principe, Self-organised direction aware data partitioning algorithm,
% Information Sciences, vol.423, pp. 80-95 , 2018.
%==========================================================================================================
%% Please cite the paper above if this code helps.
%% For any queries about the code, please contact Prof. Plamen P. Angelov and Dr. Xiaowei Gu
%% {p.angelov,x.gu3}@lancaster.ac.uk
%% Programmed by Xiaowei Gu
function [Output]=SelfOrganisedDirectionAwareDataPartitioning(Input,Mode)
    %% Output:  C            -   Identified prototypes
    %           IDX          -   Data cloud label (Clustering label, classes) of the data samples
    %           SystemParams -   Meta-parameters of the identified direction-aware planes, important for the subsequent processing with the evolving extension
%% Offline SODA data partitioning algorithm
if strcmp(Mode,'Offline')==1
    %% Input:   Input.StaticData -  input data;
    %           Input.GridSize         -  grid size
    data=Input.StaticData;
    [L,W]=size(data);
    N=Input.GridSize;
    [X1,AvD1,AvD2,grid_trad,grid_angl]=grid_set(data,N); % Create a NxN grid net to divide whole direction-aware space
    [GD,Uniquesample,Frequency]=Globaldensity_Calculator(data); % Calculate the global density at each data sample
    [BOX,BOX_miu,BOX_X,BOX_S,BOXMT,NB]=chessboard_division(Uniquesample,GD,grid_trad,grid_angl); % Project all the unique data samples into the direction-aware planes
    [Center,ModeNumber]=ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,grid_trad,grid_angl); % Identify the local maxima of global density and use the respective samples as the prototypes
    [Members,Membernumber,Membership,IDX]=cloud_member_recruitment(ModeNumber,Center,data,grid_trad,grid_angl);% Assign the data samples to the nearest prototypes as Voronoi Tessellation
    Boxparameter.BOX=BOX;
    Boxparameter.BOX_miu=BOX_miu;
    Boxparameter.BOX_S=BOX_S;
    Boxparameter.NB=NB;
    Boxparameter.XM=X1;
    Boxparameter.L=L;
    Boxparameter.AvM=AvD1;
    Boxparameter.AvA=AvD2;
    Boxparameter.GridSize=N;
end
%% Evolving extension of the SODA data partitioning algorithm
%% Input: Input.StreamingData -  input data;
%         Input.AllData       -  the whole dataset, for creating the final results with the identified prototypes
%         Input.Boxparameters -  the offline prime;
if strcmp(Mode,'Evolving')==1
    Data2=Input.StreamingData;
    data=Input.AllData;
    Boxparameters=Input.SystemParams;
    BOX=Boxparameters.BOX;
    BOX_miu=Boxparameters.BOX_miu;
    BOX_S=Boxparameters.BOX_S;
    XM=Boxparameters.XM;
    AvM=Boxparameters.AvM;
    AvA=Boxparameters.AvA;
    N=Boxparameters.GridSize;
    NB=Boxparameters.NB;
    L1=Boxparameters.L;
    L2=size(Data2,1);
    for k=1:1:L2
        [XM,AvM,AvA]=data_standardization(Data2(k,:),XM,AvM,AvA,k+L1);
        interval1=sqrt(2*(XM-AvM*AvM'))/N;
        interval2=sqrt(1-AvA*AvA')/N;
        [BOX,BOX_miu,BOX_S,NB]=Chessboard_online_division(Data2(k,:),BOX,BOX_miu,BOX_S,NB,interval1,interval2);
        [BOX,BOX_miu,BOX_S,NB]=Chessboard_online_merge(BOX,BOX_miu,BOX_S,NB,interval1,interval2);
    end
    [BOXG]=Chessboard_globaldensity(BOX_miu,BOX_S,NB);
    [Center,ModeNumber]=ChessBoard_online_projection(BOX_miu,BOXG,NB,interval1,interval2);
    [Members,Membernumber,~,IDX]=cloud_member_recruitment(ModeNumber,Center,data,interval1,interval2);
    Boxparameter.BOX=BOX;
    Boxparameter.BOX_miu=BOX_miu;
    Boxparameter.BOX_S=BOX_S;
    Boxparameter.NB=NB;
    Boxparameter.L=L1+L2;
    Boxparameter.AvM=AvM;
    Boxparameter.AvA=AvA;
end
Output.C=Center;
Output.IDX=IDX;
Output.SystemParams=Boxparameter;
end
function [X1,AvD1,AvD2,grid_trad,grid_angl]=grid_set(data,N)
[~,W]=size(data);
AvD1=mean(data);
X1=mean(sum(data.^2,2));
grid_trad=sqrt(2*(X1-AvD1*AvD1'))/N;
Xnorm = sqrt(sum(data.^2, 2));
data = data ./ Xnorm(:,ones(1,W));
seq=find(isnan(data));
data(seq,:)=ones(length(seq),W);
AvD2=mean(data);
grid_angl=sqrt(1-AvD2*AvD2')/N;
end
function [GD,Uniquesample,Frequency]=Globaldensity_Calculator(data)
[Uniquesample,J,K]=unique(data,'rows');
Frequency = histc(K,1:numel(J));
[uspi1]=pi_calculator(Uniquesample,'euclidean');
sum_uspi1=sum(uspi1);
Density_1=uspi1./sum_uspi1;
[uspi2]=pi_calculator(Uniquesample,'cosine');
sum_uspi2=sum(uspi2);
Density_2=uspi1./sum_uspi2;
GD=(Density_2+Density_1).*Frequency;
[~,Index]=sort(GD,'descend');
GD=GD(Index);
Uniquesample=Uniquesample(Index,:);
Frequency=Frequency(Index);
end
function [Box,BOX_miu,BOX_X,BOX_S,BOXMT,NB]=chessboard_division(Uniquesample,MMtypicality,interval1,interval2)
[L,~]=size(Uniquesample);
Box=Uniquesample(1,:);
BOX_miu=Uniquesample(1,:);
BOX_S=1;
BOX_X=sum(Uniquesample(1,:).^2);
NB=1;
BOXMT=MMtypicality(1);
for i=2:1:L
    distance=zeros(NB,2);
    distance(:,1)=pdist2(Uniquesample(i,:),BOX_miu,'euclidean');
    distance(:,2)=sqrt(pdist2(Uniquesample(i,:),BOX_miu,'cosine'));
    SQ=find(distance(:,1)<interval1&distance(:,2)<interval2);
    COUNT=length(SQ);
    if COUNT==0
        Box=[Box;Uniquesample(i,:)];
        NB=NB+1;
        BOX_S=[BOX_S;1];
        BOX_miu=[BOX_miu;Uniquesample(i,:)];
        BOX_X=[BOX_X;sum(Uniquesample(i,:).^2)];
        BOXMT=[BOXMT;MMtypicality(i)];
    end
    if COUNT>=1
        DIS=distance(SQ,1)/interval1+distance(SQ,2)/interval2;
        [~,b]=min(DIS);
        BOX_S(SQ(b))=BOX_S(SQ(b))+1;
        BOX_miu(SQ(b),:)=(BOX_S(SQ(b))-1)/BOX_S(SQ(b))*BOX_miu(SQ(b),:)+Uniquesample(i,:)/BOX_S(SQ(b));
        BOX_X(SQ(b))=(BOX_S(SQ(b))-1)/BOX_S(SQ(b))* BOX_X(SQ(b))+sum(Uniquesample(i,:).^2)/BOX_S(SQ(b));
        BOXMT(SQ(b))=BOXMT(SQ(b))+MMtypicality(i);
    end
end
end
function [Centers,ModeNumber]=ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,Internval1,Internval2)
Centers=[];
n=2;
ModeNumber=0;
distance1=squareform(pdist(BOX_miu,'euclidean'));
distance2=sqrt(squareform(pdist(BOX_miu,'cosine')));
for ii=1:1:NB
    seq=find(distance1(ii,:)<n*(Internval1)&distance2(ii,:)<n*Internval2);
    Chessblocak_typicality=BOXMT(seq);
    if max(Chessblocak_typicality)==BOXMT(ii)
        Centers=[Centers;BOX_miu(ii,:)];
        ModeNumber=ModeNumber+1;
    end
end
end
function [Members,Membernumber,Membership,B]=cloud_member_recruitment(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl)
[L,W]=size(Uniquesample);
Membership=zeros(L,ModelNumber);
Members=zeros(L,ModelNumber*W);
Count=zeros(1,ModelNumber);
distance3=pdist2(Uniquesample,Center_samples,'euclidean')/grid_trad+sqrt(pdist2(Uniquesample,Center_samples,'cosine'))/grid_angl;
[~,B]=min(distance3,[],2);
for ii=1:1:ModelNumber
    seq=find(B==ii);
    Count(ii)=length(seq);
    Membership(1:1:Count(ii),ii)=seq;
    Members(1:1:Count(ii),1+W*(ii-1):1:W*ii)=Uniquesample(seq,:);
end
Membernumber=Count;
end
function [uspi]=pi_calculator(Uniquesample,mode)
[UN,W]=size(Uniquesample);
if strcmp(mode,'euclidean')==1
    AA1=mean(Uniquesample,1);
    X1=(sum(sum(Uniquesample.^2)))/UN;
    DT1=X1-AA1*AA1';
    uspi=sum((Uniquesample-repmat(AA1,size(Uniquesample,1),1)).^2,2)+DT1;
end
if strcmp(mode,'cosine')==1
    Xnorm = sqrt(sum(Uniquesample.^2, 2));
    Uniquesample1 = Uniquesample ./ Xnorm(:,ones(1,W));
    AA2=mean(Uniquesample1,1);
    X2=1;
    DT2=X2-AA2*AA2';
    uspi=sum((Uniquesample1-repmat(AA2,size(Uniquesample,1),1)).^2,2)+DT2;
end
end
function [Box_new,BOX_miu_new,BOX_S_new,NB_new]=Chessboard_online_division(data,Box,BOX_miu,BOX_S,NB,intervel1,intervel2)
distance=zeros(NB,2);
COUNT=0;
SQ=[];
for i=1:1:NB
    distance(i,1)=pdist([BOX_miu(i,:);data],'euclidean');
    distance(i,2)=sqrt(pdist([BOX_miu(i,:);data],'cosine'));
    if distance(i,1)<intervel1&&distance(i,2)<intervel2
        COUNT=COUNT+1;
        SQ=[SQ;i];
    end
end
if COUNT==0
    Box_new=[Box;data];
    NB_new=NB+1;
    BOX_S_new=[BOX_S;1];
    BOX_miu_new=[BOX_miu;data];
end
if COUNT>=1
    DIS=zeros(COUNT,1);
    for j=1:1:COUNT
        DIS(j)=distance(SQ(j),1)+distance(SQ(j),2);
    end
    [a,b]=min(DIS);
    Box_new=Box;
    NB_new=NB;
    BOX_S_new=BOX_S;
    BOX_miu_new=BOX_miu;
    BOX_S_new(SQ(b))=BOX_S(SQ(b))+1;
    BOX_miu_new(SQ(b),:)=(BOX_S(SQ(b)))/BOX_S_new(SQ(b))*BOX_miu(SQ(b),:)+data/BOX_S_new(SQ(b));
end
end
function [Box,BOX_miu,BOX_S,NB]=Chessboard_online_merge(Box,BOX_miu,BOX_S,NB,intervel1,intervel2)
threshold1=intervel1/2;
threshold2=intervel2/2;
NB1=0;
while NB1~=NB
    CC=0;
    NB1=NB;
    for ii=1:1:NB
        seq1=[1:1:ii-1,ii+1:1:NB];
        distance1=pdist2(BOX_miu(ii,:),BOX_miu(seq1,:),'euclidean');
        distance2=sqrt(pdist2(BOX_miu(ii,:),BOX_miu(seq1,:),'cosine'));
        for jj=1:1:NB-1
            if distance1(1,jj)<threshold1&&distance2(1,jj)<threshold2
                CC=1;
                NB=NB-1;
                Box(ii,:)=[];
                BOX_miu(seq1(jj),:)=BOX_miu(seq1(jj),:)*BOX_S(seq1(jj))/(BOX_S(seq1(jj))+BOX_S(ii))+BOX_miu(ii,:)*BOX_S(ii)/(BOX_S(seq1(jj))+BOX_S(ii));
                
                BOX_S(seq1(jj))=BOX_S(seq1(jj))+BOX_S(ii);
                BOX_miu(ii,:)=[];
                BOX_S(ii)=[];
                
                break
            end
        end
        if CC==1
            break
        end
    end
end
end
function [X_global_new,mean_global_new,mean_global2_new]=data_standardization(data,X_global,mean_global,mean_global2,k)
mean_global_new=(k-1)/k*mean_global+data/k;
X_global_new=(k-1)/k*X_global+sum(data.^2)/k;
mean_global2_new =(k-1)/k*mean_global2+data/k/(sqrt(sum(data.^2)));
end
function [Hyper_GD]=Chessboard_globaldensity(Hypermean,HyperSupport,NH)
[uspi1]=pi_calculator(Hypermean,'euclidean');
sum_uspi1=sum(uspi1);
Density_1=uspi1./sum_uspi1;
[uspi2]=pi_calculator(Hypermean,'cosine');
sum_uspi2=sum(uspi2);
Density_2=uspi1./sum_uspi2;
Hyper_GD=(Density_2+Density_1).*HyperSupport;
% Hyper_GD=zeros(NH,1);
% for i=1:1:NH
%     Hyper_GD(i)=(Density_2(i)+Density_1(i))*HyperSupport(i);
% end
end
function [Centers,ModeNumber]=ChessBoard_online_projection(BOX_miu,BOXMT,NB,intervel1,intervel2)
Centers=[];
ModeNumber=0;
n=2;
for ii=1:1:NB
    Reference=BOX_miu(ii,:);
    distance1=zeros(NB,1);
    distance2=zeros(NB,1);
    for i=1:1:NB
        distance1(i)=sqrt(sum((Reference-BOX_miu(i,:)).^2));
        distance2(i)=sqrt(pdist([Reference;BOX_miu(i,:)],'cosine'));
    end
    Chessblocak_typicality=[];
    for i=1:1:NB
        if distance1(i)<n*intervel1&&distance2(i)<n*intervel2
            Chessblocak_typicality=[Chessblocak_typicality;BOXMT(i)];
        end
    end
    if max(Chessblocak_typicality)==BOXMT(ii)
        Centers=[Centers;Reference];
        ModeNumber=ModeNumber+1;
    end
end
end
