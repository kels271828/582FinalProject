function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
% X, y, lambda) computes the cost and gradient of the neural network. The
% parameters for the neural network are "unrolled" into the vector
% nn_params and need to be converted back into the weight matrices. 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

%% Feedforward computation

a1 = [ones(1, m) ; X'];
z2 = Theta1 * a1;
a2 = [ones(1, m) ; sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% set up Y matrix
Y = zeros(num_labels, m);
for k = 1:num_labels 
    Y(k, :) = (y == k);
end

% compute the cost
J = 1 / m * sum(diag(-Y' * log(a3) - (1 - Y') * log(1 - a3)));

%% Backpropogation

% compute delta matrices
delta3 = a3 - Y;
temp = Theta2' * delta3;
delta2 = temp(2:end, :) .* sigmoidGradient(z2);

% compute gradients
Theta1_grad = 1 / m * delta2 * a1';
Theta2_grad = 1 / m * delta3 * a2';

% add regularization to cost function
for j = 1:hidden_layer_size
    for k = 1:input_layer_size
        J = J + lambda / (2 * m) * Theta1(j, k+1)^2;
    end
end
for j = 1:num_labels
    for k = 1:hidden_layer_size
        J = J + lambda / (2 * m) * Theta2(j, k+1)^2;
    end
end

% add regularization to gradients
A1 = ones(size(Theta1_grad));
A1(:, 1) = zeros(hidden_layer_size, 1);
Theta1_grad = Theta1_grad + lambda / m * A1 .* Theta1;

A2 = ones(size(Theta2_grad));
A2(:, 1) = zeros(num_labels, 1);
Theta2_grad = Theta2_grad + lambda / m * A2 .* Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
