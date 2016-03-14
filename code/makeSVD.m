% Calculate SVD of data matrix

clear all; close all; clc;

%--------------------------------------------------------------------------
% BBC Dataset Contents
%--------------------------------------------------------------------------
% bbc.mtx     : Original term frequencies stored in a sparse data matrix in
%               Matrix Market format
% bbc.terms   : List of content-bearing terms in the corpus, with each line
%               corresponding to a row of the sparse data matrix
% bbc.docs    : List of document identifiers, with each line corresponding
%               to a column of the sparse data matrix
% bbc.classes : Assignment of documents to natural classes, with each line
%               corresponding to a document
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Matrix Market Functions
%--------------------------------------------------------------------------
% [rows,cols,entries,rep,field,symm] = mminfo(filename)
% [A,rows,cols,entries] = mmread(filename)
% [err] = mmwrite(filename,A,comment,field,precision)
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Matrix setup
%--------------------------------------------------------------------------
%    1:510  Business      (510)
%  511:896  Entertainment (386)
%  897:1313 Politics      (417)
% 1314:1824 Sports        (511)
% 1825:2225 Tech          (401)
%--------------------------------------------------------------------------

% 1 = raw dataset
% 2 = robust PCA dataset
dataset = 1;

% fork based on which dataset to process
if dataset == 1
    [rawA,~,~,~] = mmread('../data/original/bbc/bbc.mtx');
    Acount = rawA;
elseif dataset == 2
    load('../data/robust/L.mat');
    Acount = L;
end

%% Splitting training data out

clc
restrictions = [1 510; 511 896; 897 1313; 1314 1824; 1825 2225];
numPerClass = diff(restrictions,[],2)+1;

%% preprocessing step

% divide by column sums
cols = size(Acount,2);
Afreq = Acount * spdiags(1./sum(Acount,1).',0,cols,cols);

% subtract row means
mu = mean(Afreq,2);
Afreq = Afreq - repmat(mu,[1 cols]);

%% calculate svd

k = 10; % number of singular values needed
[U,Sigma,V] = svds(Afreq,k);

%% save output

outdir = '../data/';
if dataset == 1
    outdir = [outdir 'original'];
elseif dataset == 2
    outdir = [outdir 'robust'];
end
outdir = [outdir '/'];

Vclasses = mat2cell(V,numPerClass(1:end),k);

save([outdir 'A.mat'],'Afreq');
save([outdir 'U.mat'],'U');
save([outdir 'Sigma.mat'],'Sigma');
save([outdir 'V.mat'],'V');
save([outdir 'Vclasses.mat'],'Vclasses');
save([outdir 'mu.mat'],'mu');

%% plot singular values

plot(diag(Sigma)/sum(diag(Sigma)),'o')
xlabel('Singular Value','FontSize',14)
ylabel('Percent Energy','FontSize',14)
title('Ten Largest Singular Values','FontSize',14)
