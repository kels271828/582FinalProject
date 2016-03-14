function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
% logisitc regression classifiers and returns each of these classifiers
% in a matrix all_theta, where the i-th row of all_theta corresponds 
% to the classifier for label i

% Initialize variables
m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);
X = [ones(m, 1) X];

% Train 5 regularized logistic regression classifiers
for c = 1:num_labels
    
    % Set Initial theta
    initial_theta = zeros(n + 1, 1);

    % Set options for fminumc
    options = optimset('GradObj', 'on', 'MaxIter', 50);

    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost
    [theta] = fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
              initial_theta, options);
    all_theta(c, :) = theta(:);        
end

end
