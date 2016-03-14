% Find n most important words for each mode

clear all; close all; clc;

% load principal component word vectors
load('../data/original/U.mat')

% find indices for important words
n = 25;
termIdx = zeros(n,10);
values = zeros(n,10);
for i = 1:10
    [val,idx] = sort(abs(U(:,i)),'descend');
    termIdx(:,i) = idx(1:n);
    values(:,i) = val(1:n);
end
    
% load word vector
bbcTerms = fopen('../data/original/bbc/bbcTerms.txt');
terms = textscan(bbcTerms,'%s');
fclose(bbcTerms);
terms = terms{1}; terms = terms(9:end);

% print out important words
importantTerms = zeros(10*n,1);
for i = 1:4
    fprintf('Top %d terms for mode %d\n',n,i)
    for j = 1:n
        fprintf('%s \n',terms{termIdx(j,i)})
    end
    fprintf('\n')
end