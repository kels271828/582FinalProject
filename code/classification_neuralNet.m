%% Neural Network Learning

clear all; close all; clc;

% 1 = raw dataset
% 2 = robust PCA dataset
dataset = 1;

% fork based on which dataset to process
if dataset == 1
    dir = '../data/original/';
elseif dataset == 2
    dir = '../data/robust/';
end

% load data
load([dir 'U.mat']);
load([dir 'Sigma.mat']);
load([dir 'V.mat']);
load([dir 'Vclasses.mat']);
load([dir 'verify.mat']);

% project the data
Ptrain = (Sigma*V')';
Ptest = (U' * cell2mat(Afreq_verify))';

% create label vectors
ytrain = cellfun(@(n) n*ones(size(Vclasses{n},1),1),num2cell(1:5),'uniformoutput',0);
ytrain = cell2mat(ytrain.');
ytest = cellfun(@(n) n*ones(size(Afreq_verify{n},2),1),num2cell(1:5),'uniformoutput',0);
ytest = cell2mat(ytest.');

%% compare accuracy for different numbers of principal components

hidden_layer_size = 25;
num_labels = 5;
options = optimset('MaxIter', 200); % use more iterations for better accuracy
lambda = 1; % regularization parameter
accuracy = zeros(10,2);
for r = 1:10;
    input_layer_size = r;
    Ptrain_r = Ptrain(:,1:r);
    Ptest_r = Ptest(:,1:r);

    % initialize parameters
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    % unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, Ptrain_r, ytrain, lambda);

    % train parameters
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    % obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % trainind data accuracy
    predTrain = predict(Theta1, Theta2, Ptrain_r);
    accuracy(r,1) = mean(double(predTrain == ytrain)) * 100;
    
    % cross validation data accuracy
    predTest = predict(Theta1, Theta2, Ptest_r);
    accuracy(r,2) = mean(double(predTest == ytest)) * 100;
end

%% plot accuracy vs. rank

clc
hold on
plot(1:10,accuracy(:,1),'r','linewidth',2);
plot(1:10,accuracy(:,2),'b','linewidth',2);
ylim([40 100]);
if dataset == 1
    title('Traditional PCA','FontSize',14)
else
    title('Robust PCA','FontSize',14)
end
legend('Training Accuracy','Testing Accuracy','Location','SouthEast');
xlabel('Number of Principal Components','FontSize',14)
ylabel('Classification Accuracy','FontSize',14)
