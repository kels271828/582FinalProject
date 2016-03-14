%% Classification by nearest-mean

clear all; clc;

% 1 = raw dataset
% 2 = robust PCA dataset
dataset = 1;

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
shortLabels = {'B','E','P','S','T'};
colors = {[0 .5 1],[1 .5 0],[0 .8 0],[1 0 .5],[.5 0 .75]};
lightColors = cellfun(@(c) .5+.5*c,colors,'uniformoutput',0);

%% classification by nearest-mean using r principal components

accuracy = zeros(10,6);
for r = 1:10; % r = rank
    Sigma2 = Sigma(1:r,1:r);
    
    % average projections from training data
    avg = zeros(r,5);
    for i = 1:5
        avg(:,i) = mean(Sigma2*Vclasses{i}(:,1:r)',2);
    end
    
    % classify verification data
    correct = zeros(5,1);
    verifyCount = zeros(5,1);
    for ii = 1:5 % classes to test from
        verifyCount(ii) = size(Afreq_verify{ii},2);
        for jj=1:size(Afreq_verify{ii},2)
            % Compute projection for particular test article
            testarticle = Afreq_verify{ii}(:,jj);
            P = U(:,1:r)' * testarticle;
            discrepancies = avg - repmat(P,[1 5]);
            [~,ind] = min(sum(discrepancies.^2,1));
            if ind==ii
                correct(ii) = correct(ii)+1;
            end
        end
    end
    accuracy(r,1:5) = correct./verifyCount;
    accuracy(r,6) = sum(correct)/sum(verifyCount);
end

%% plot results for nearest-mean

clf
hold on
myplots = [];
myplots(1) = plot(1:10,accuracy(:,6),'ko:','markerfacecolor','w');
for k=1:5
    myplots(k+1) = plot(1:10,accuracy(:,k),'ro-','color',colors{k},'markerfacecolor','w');
end
legtext = {'Average',labels{:}};
legend(myplots,legtext,'location','southeast');
if dataset == 1
    title('Traditional PCA','FontSize',14)
else
    title('Robust PCA','FontSize',14)
end
xlabel('Number of Principal Components','FontSize',14)
ylabel('Classification Accuracy','FontSize',14)
