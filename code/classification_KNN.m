% Classification using K Nearest Neighbors

clear all; close all; clc;

% 1 = raw dataset
% 2 = robust PCA dataset
dataset = 2;

% fork based on which dataset to process
if dataset == 1
    dir = '../data/original/';
    load([dir 'Sigma.mat']);
    load([dir 'U.mat']);
    load([dir 'V.mat']);
    load([dir 'verify.mat']);
    load([dir 'Vclasses.mat'])
elseif dataset == 2
    dir = '../data/robust/';
    load([dir 'Sigma.mat']);
    load([dir 'U.mat']);
    load([dir 'V.mat']);
    load([dir 'verify.mat']);
    load([dir 'Vclasses.mat'])
end

% class indices and labels
restrictions = [1 510; 511 896; 897 1313; 1314 1824; 1825 2225];
classes = cellfun(@(x) {x},num2cell(restrictions),'uniformoutput',0);
labels = {'Business','Entertainment','Politics','Sports','Tech'};
colors = {[0 .5 1],[1 .5 0],[0 .8 0],[1 0 .5],[.5 0 .75]};
lightColors = cellfun(@(c) .5+.5*c,colors,'uniformoutput',0);

%% KNN on cross validation data

Kmaxneighbors = 20;
accuracy = zeros(10,5,Kmaxneighbors);
for r=1:10
    Sigmar = Sigma(1:r,1:r);
    
    trainingprojections = cellfun(@(c) c(:,1:r).',Vclasses,'uniformoutput',0);
    trainingprojections = [trainingprojections{:}];
    trainingprojections = Sigmar*trainingprojections;
    
    trainingclasses = cellfun(@(cind) cind*ones(1,size(Vclasses{cind},1)), num2cell(1:5),'uniformoutput',0);
    trainingclasses = [trainingclasses{:}].';
    
    testprojections = [Afreq_verify{:}];
    P = U(:,1:r)' * testprojections;
    correctclasses = cellfun(@(cind) cind*ones(1,size(Afreq_verify{cind},2)), num2cell(1:5),'uniformoutput',0);
    correctclasses = [correctclasses{:}].';
    for kneighbors=1:Kmaxneighbors
        
        inds = knnsearch(trainingprojections.',P.','k',kneighbors);
        reportedclasses = trainingclasses(inds);
        reportedclasses = mode(reportedclasses,2);
        
        for i=1:5
            accuracy(r,i,kneighbors) = sum((correctclasses==i) & (reportedclasses==i))/sum(correctclasses==i);
        end
    end
end

%% Plot accuracy versus rank, for fixed number of neighbors

figure(), hold on
myplots = [];
for k=15
    cla
    for i=1:5
        plot(1:10,accuracy(:,i,k),'ro-','color',colors{i},'markerfacecolor','w');
    end
    ylim([0 1]);
end
xlabel('Number of Principal Components','FontSize',14)
ylabel('Classification Accuracy','FontSize',14)

%% Plot accuracy versus number of neighbors, for fixed rank

figure(), hold on
for r=3
    for i=1:5
        plot(1:Kmaxneighbors,reshape(accuracy(r,i,:),[],1),'ro-','color',colors{i},'markerfacecolor','w');
    end
end
ylim([0 1]);
xlabel('Number of Nearest Neighbors','FontSize',14)
ylabel('Classification Accuracy','FontSize',14)
