%% Ant Colony Optimization (ACO) Image Feature Extraction Method
% Feature Extraction and Feature Selection are two different tasks. Feature
% Extraction is initial and vital step, but feature selection is optional.
% There are lots of evolutionary feature selection code are online for
% MATLAB but not feature extraction, especially for image. This code extracts
% features out of 10 classes of images with Ant Colony Optimization (ACO)
% evolutionary algorithm and compared it with extracted features using
% SURF with KNN classifier. Dataset is consists of 100 samples of small
% objects in 10 classes. You can use your data but labeling is done manually
% which you have to change it. following parameters are so important which
% you have to play with them in order to get desired results. Parameters
% are: 'nf', 'MaxIt', 'nAnt', knn classifier neighbors and number of hidden
% layers in "TrainNN.m" file. Feel free to contact me:
% Email: mosavi.a.i.buali@gmail.com
% Author: Seyed Muhammad Hossein Mousavi
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi
% This code is part of the following project, so if you used the code,
% please cite below paper:
% Mousavi, Seyed Muhammad Hossein, S. Younes MiriNezhad, and Mir Hossein Dezfoulian.
% "Galaxy gravity optimization (GGO) an algorithm for optimization, inspired by comets
% life cycle." 2017 Artificial Intelligence and Signal Processing Conference (AISP).
% IEEE, 2017.
% Hope it help you (Be Happy :)

%% Lets Go...
clear;
warning('off');
% Loading The Dataset
path='data';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
disp(['Loading Image Number :   ' num2str(i) ]);end;
% Color to Gray Conversion
for i = 1 : filesnumber(1,1)
gray{i}=rgb2gray(images{i}); 
disp(['Gray Conversion Image Number :   ' num2str(i) ]);end;
% Resizing Images
for i = 1 : filesnumber(1,1)
resized{i}=imresize(gray{i}, [32 32]); 
disp(['Resizing Image Number :   ' num2str(i) ]);end;
% Uint8 Image to Double Image Conversion
for i = 1 : filesnumber(1,1)
adj{i}=im2double(imadjust(resized{i})); 
disp(['Converting Images to Double Type :   ' num2str(i) ]);end;
% Converting Image Matrix to Vector (n*n to 1*n)
% This Step is Vital
for i = 1 : filesnumber(1,1)
vector(i,:)=reshape(adj{i},1,[]);
disp(['Image Matrix to Vector Image :   ' num2str(i) ]);end;
% Labeling (10 classes)- (it is done manually-change it for your data)
vector2=vector;
vector2(1:10,1025)=1;
vector2(11:20,1025)=2;
vector2(21:30,1025)=3;
vector2(31:40,1025)=4;
vector2(41:50,1025)=5;
vector2(51:60,1025)=6;
vector2(61:70,1025)=7;
vector2(71:80,1025)=8;
vector2(81:90,1025)=9;
vector2(91:100,1025)=10;

%% ACO Feature Extraction Starts
% Preparing Dataset
sizdet=size(vector2);
x=vector';
t=vector2(:,sizdet(1,2))';
nx=sizdet(1,2)-1;
nt=1;
nSample=sizdet(1,1);
% Converting Table to Struct
data.x=x;
data.t=t;
data.nx=nx;
data.nt=nt;
data.nSample=nSample;

% 'nf' is the number of features (depends on you)
nf=15;   % Desired Number of Selected Features
%
CostFunction=@(q) FeatureExtCost(q,nf,data);    
nVar=data.nx;

%% ACO Parameters
MaxIt=15;      % Maximum Number of Iterations
nAnt=3;        % Number of Ants (Population Size)
Q=1;
tau0=1;	        % Initial Phromone
alpha=1;        % Phromone Exponential Weight
beta=1;         % Heuristic Exponential Weight
rho=0.05;       % Evaporation Rate
%
eta=ones(nVar,nVar);        % Heuristic Information Matrix
tau=tau0*ones(nVar,nVar);   % Phromone Matrix
BestCost=zeros(MaxIt,1);    % Array to Hold Best Cost Values
% Empty Ant
empty_ant.Tour=[];
empty_ant.Cost=[];
empty_ant.Out=[];
% Ant Colony Matrix
ant=repmat(empty_ant,nAnt,1);
% Best Ant
BestAnt.Cost=inf;
%% ACO Starts
for it=1:MaxIt
% Move Ants
for k=1:nAnt
            ant(k).Tour=randi([1 nVar]);
for l=2:nVar
            i=ant(k).Tour(end);
            P=tau(i,:).^alpha.*eta(i,:).^beta;
            P(ant(k).Tour)=0;
            P=P/sum(P);
            j=RouletteWheelSelection(P);
            ant(k).Tour=[ant(k).Tour j];
end
            [ant(k).Cost, ant(k).Out]=CostFunction(ant(k).Tour);
            if ant(k).Cost<BestAnt.Cost
            BestAnt=ant(k);
end
end
    % Update Phromones
for k=1:nAnt
        tour=ant(k).Tour;
        tour=[tour tour(1)];
for l=1:nVar
            i=tour(l);
            j=tour(l+1);
            tau(i,j)=tau(i,j)+Q/ant(k).Cost;
end
end
    % Evaporation
    tau=(1-rho)*tau;
    % Store Best Cost
    BestCost(it)=BestAnt.Cost;
    % Show Iteration Information
    disp(['In Iteration Number ' num2str(it) ': Prime Cost Is = ' num2str(BestCost(it))]);
end

%% Plot Res
figure;
set(gcf, 'Position',  [450, 250, 900, 250])
plot(BestCost,'-.',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'Color',[0.9,0,0.9]);
title('Ant Colony Optimization Train')
xlabel('ACO Iteration Number','FontSize',12,...
       'FontWeight','bold','Color','m');
ylabel('ACO Best Cost Result','FontSize',12,...
       'FontWeight','bold','Color','m');
legend({'ACO Train'});

%% Ant Colony Optimization Feature Selection (Final Step)
% Extracting Data
RealData=vector2;
% Extracting Labels
RealLbl=RealData(:,end);
FinalFeaturesInd=BestAnt.Out.S;
% Sort Features
FFI=sort(FinalFeaturesInd);
% Select Final Features
ACO_Features=RealData(:,FFI);
% Adding Labels
ACO_Features(:,end+1)=RealLbl;

%% Extract SURF Features for Comparison With ACO Features
lbl=ACO_Features(:,end);
imset = imageSet('surf','recursive'); 
% Create a bag-of-features from the image database
bag = bagOfFeatures(imset,'VocabularySize',nf,'PointSelection','Detector');
% Encode the images as new features
surf = encode(bag,imset);
surf(:,end+1)=lbl;

%% Classification
% KNN On ACO Features
lblknn=ACO_Features(:,end); lbllbl=lblknn;
dataknn=ACO_Features(:,1:end-1);
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',9,'Standardize',1);
rng(1); % For reproducibility
knndat = crossval(Mdl);
classErroraco = kfoldLoss(knndat);
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl); preknn=predictedknn;
ctknnaco=0;
for i = 1 : nSample(1,1)
if lblknn(i) ~= predictedknn(i)
    ctknnaco=ctknnaco+1;
end;
end;
finknn=ctknnaco*100/ nSample;
KNN_ACO=(100-finknn)-classErroraco;

% KNN On SURF Features
lblknn=surf(:,end);
dataknn=surf(:,1:end-1);
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',9,'Standardize',1);
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat);
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl);
ctknnsurf=0;
for i = 1 : nSample(1,1)
if lblknn(i) ~= predictedknn(i)
    ctknnsurf=ctknnsurf+1;
end;
end;
finknn=ctknnsurf*100/ nSample;
KNN_SURF=(100-finknn)-classError;

% Confusion Matrixes
figure
set(gcf, 'Position',  [150, 150, 1000, 350])
subplot(1,2,1)
cmknn = confusionchart(lbllbl,preknn);
cmknn.Title = (['KNN on SURF Features =  ' num2str(KNN_ACO) '%']);
cmknn.RowSummary = 'row-normalized';
cmknn.ColumnSummary = 'column-normalized';
subplot(1,2,2)
cmknn1 = confusionchart(lblknn,predictedknn);
cmknn1.Title = (['KNN on ACO Features =  ' num2str(KNN_SURF) '%']);
cmknn1.RowSummary = 'row-normalized';
cmknn1.ColumnSummary = 'column-normalized';
% Final Accuracy 
fprintf('The (KNN Accuracy on SURF Features) is =  %0.4f.\n',KNN_SURF)
fprintf('The (KNN Accuracy on ACO Features) is =  %0.4f.\n',KNN_ACO)

