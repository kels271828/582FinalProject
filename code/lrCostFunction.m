function [J, grad] = lrCostFunction(theta, X, y, lambda)
% J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
% theta as the parameter for regularized logistic regression and the
% gradient of the cost w.r.t. to the parameters. 

% Initialize variables
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta
J = 1 / m * (-y' * log(sigmoid(X * theta)) - ...
    (1 - y)' * log(1 - sigmoid(X * theta))) + ...
    lambda / (2 * m) * (theta(2:end)' * theta(2:end));

% Compute the partial derivatives w.r.t each parameter in theta
A = speye(size(theta, 1));
A(1, 1) = 0;
grad = 1 / m * X' * (sigmoid(X * theta) - y) + ...
       lambda / m * A * theta;
grad = grad(:);

end
