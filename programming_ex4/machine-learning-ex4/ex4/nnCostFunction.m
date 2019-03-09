function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(size(X,1),1) X];

z2 = X*Theta1';

a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

A = zeros(size(X,1), num_labels);
for i=1:size(X,1);
A(i, y(i)) = 1;
end;
J = sum((-A.*log(a3) - (1-A).*log(1-a3))(:))/size(X,1);

J = J + (lambda/(2*m))*(sum((Theta1(:, 2:end).^2)(:)) + sum((Theta2(:, 2:end).^2)(:)));


%先按照它说的做，但我觉得应该可以不用for loop啊
for i=1:m
	%计算a3
	% // x = X(i, :)';
	% // z2 = Theta1 * x;
	% // a2 = sigmoid(z2);
	% // z2 = [1;z2];
	% // a2 = [1;a2];
	% // z3 = Theta2 * a2;
	% // a3 = sigmoid(z3);

	%算各层的误差
    delta3 = (a3(i, :)' - A(i, :)');

    delta2 = (Theta2'*delta3).*sigmoidGradient([1;z2(i, :)']);
    Theta2_grad = Theta2_grad + delta3 * a2(i,:);
    Theta1_grad = Theta1_grad + delta2(2:end) * X(i,:);
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;
re2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
re1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + re2*lambda/m;
Theta1_grad = Theta1_grad + re1*lambda/m;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
