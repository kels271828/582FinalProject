% Visualize SVD structure (training data)

clear all; close all; clc;

% 1 = raw dataset
% 2 = robust PCA dataset
dataset = 1;

% fork based on which dataset to process
if dataset == 1
    load('../data/original/U.mat')
    load('../data/original/Sigma.mat')
    load('../data/original/V.mat')
    load('../data/original/Vclasses.mat')
elseif dataset == 2
    load('../data/robust/U.mat')
    load('../data/robust/Sigma.mat')
    load('../data/robust/V.mat')
    load('../data/robust/Vclasses.mat')
end

% useful variables
percentEnergy = 100*diag(Sigma)/sum(diag(Sigma));
numWords = size(U,1); numArticles = size(V,1);
classes = {{1:510},{511:896},{897:1313},{1314:1824},{1825:2225}};
colors = {[0 .5 1],[1 .5 0],[0 .8 0],[1 0 .5],[.5 0 .75]};
lightColors = cellfun(@(c) .5+.5*c,colors,'uniformoutput',0);
labels = {'Business','Entertainment','Politics','Sports','Tech'};
shortLabels = {'B','E','P','S','T'};

%% visualize principal components (columns of U and V)

for i = 1:10
    clf, subplot(2,1,1)
    plot(U(:,i),'b.')
    xlabel('Words','FontSize',14)
    axis([0 numWords min(U(:,i)) max(U(:,i))])
    set(gca,'XLim',[0 numWords])
    title(sprintf('Singular Value #%d = %.2f Percent',i,percentEnergy(i)),'FontSize',14)
    subplot(2,1,2), hold on
    mystart = 1;
    for j = 1:5
        myend = numel(Vclasses{j}(:,1));
        plot(mystart-1+[1:myend],Vclasses{j}(:,i),'.','color',colors{j})
        mystart = mystart + myend;
    end
    plot([0 numArticles],[0 0],'k')
    xlabel('Articles','FontSize',14)
    axis([0 numArticles min(V(:,i)) max(V(:,i))])
    pause()
end


%% visualize projection onto principal components (columns of V)

clf
set(gcf,'position',[20 50 500 650],'paperpositionmode','auto')
enums = {'First','Second','Third','Fourth','Fifth'};
for i = 1:5
    axes('position',[.175 .96-i*.18 .8 .16],'fontsize',12,'xtick',[])
    hold on
    myplots = [];
    mystart = 1;
    for j = 1:5
        if j > 1
            plot(mystart*[1 1],[-1 1],'linewidth',1.5,'color',.6*[1 1 1]);
        end
        mynum = numel(Vclasses{j}(:,1));
        myplots(j) = plot(mystart-1+[1:mynum],Vclasses{j}(:,i),'.','color',colors{j},'markersize',15);
        mystart = mystart + mynum;
    end
    plot([0 numArticles],[0 0],'k','linewidth',2)
    if i == 5
        xlabel('Articles ','FontSize',20)
    else
        set(gca,'XTick',[]);
    end
    if i == 1
        if dataset == 1
            title('Traditional PCA ','FontSize',20)
        else
            title('Robust PCA ','FontSize',20)
        end
    end
    ylabel(sprintf('PC %d ',i),'fontsize',20);
    legtext = labels;
    xlim([0 numArticles]);
    myYlim = [min(V(:,i)) max(V(:,i))];
    myYlim = myYlim + .05*diff(myYlim)*[-1 1];
    ylim(myYlim)
end
if dataset == 1
    print(gcf,'-dpng','-r300','../figures/principalcomponents_original.png');
else
    print(gcf,'-dpng','-r300','../figures/principalcomponents_robust.png');
end

%% 3D plot of projection onto three principal components

% choose 3 principal components
c = [1 3 4];

% 3D projection
figure(), hold on
for k=1:5
    P = Sigma*Vclasses{k}.';
    plot3(P(c(1),:),P(c(2),:),P(c(3),:),'.','color',colors{k});
end
if dataset == 1
    title('Traditional PCA','FontSize',14)
else
    title('Robust PCA','FontSize',14)
end
xlabel(sprintf('Principal Component %d',c(1)),'FontSize',14)
ylabel(sprintf('Principal Component %d',c(2)),'FontSize',14)
zlabel(sprintf('Principal Component %d',c(3)),'FontSize',14)
axis vis3d

%% 2D projection onto three principal components

figure()
d = [c(1) c(2); c(1) c(3); c(2) c(3)];
for i = 1:3
    % plot projections
    subplot(1,3,i), hold on
    for j = 1:5
        P = Sigma*Vclasses{j}.';
        plot(P(d(i,1),:),P(d(i,2),:),'.','color',colors{j});
    end
    
    % add average articles
    for j = 1:5
        avg = mean(Sigma*Vclasses{j}',2);
        plot(avg(d(i,1)),avg(d(i,2)),'ko',...
            'LineWidth',2,'MarkerFaceColor',lightColors{j},'MarkerSize',10)
        axis square
    end
    if i == 2
        if dataset == 1
            title('Traditional PCA','FontSize',14)
        else
            title('Robust PCA','FontSize',14)
        end
    end
    xlabel(sprintf('Principal Component %d',d(i,1)),'FontSize',14)
    ylabel(sprintf('Principal Component %d',d(i,2)),'FontSize',14)
end
