clear all; close all; clc;

[S,~,~,~] = mmread('../data/robust/S.dat');
%%
spar = sparse(S);
match = spar~=0;
[row,col] = find(match);

nonzeros = [row, col, full(spar(match))];
%%
clc
myfile = fopen('../data/original/bbc/bbc.terms','r');
terms = textscan(myfile,'%s');
terms = terms{1};
sparse_terms = cell(252,5);
sparse_terms(:,1) = terms(row);
sparse_terms(:,2) = num2cell(col);
sparse_terms(:,3) = num2cell(nonzeros(:,3));
%%
test = nonzeros(:,1);
word_freq = zeros(252,1);
for k = 1:length(test)
    word_freq(k) = sum(test == test(k));
end
sparse_terms(:,4) = num2cell(word_freq);

cat_inds = [510, 896, 1313, 1824, 2225]; %last of each
labels = {'Business','Entertainment','Politics','Sports','Tech'};

for k =1:252
    article = cell2mat(sparse_terms(k,2));
    cat = 6 - sum(article <= cat_inds);
    sparse_terms(k,5) = labels(cat);
end
%%
[res,I ] = sort(cell2mat(sparse_terms(:,3)));
sorted_sparse_terms = sparse_terms(I,:);

sample = sorted_sparse_terms(235:end,:);




