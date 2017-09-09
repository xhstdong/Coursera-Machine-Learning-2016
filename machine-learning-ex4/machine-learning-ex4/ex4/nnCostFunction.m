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
% for our 2 layer neural network. It was vectorized elsewhere
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

%change training results y from m by 1 to m by n
n=max(y);
Y=zeros([m n]);
for i=1:m
    Y(i,y(i))=1;
end

%forward propagation to find cost function
%first layer
a_1=[ones([m 1]) X]; %X is a_1
z_2=a_1*Theta1';
a_2=sigmoid(z_2); %this is a_2

%second layer
a_2=[ones([m 1]) a_2];
z_3=a_2*Theta2';
a_3=sigmoid(z_3); %this is a_3

J=sum(sum(-log(a_3).*Y-(ones([m n])-Y).*log(ones([m n])-a_3)))/m;
%regularization
Theta1_trunc=Theta1(:,2:end);
Theta2_trunc=Theta2(:,2:end);
J=J+lambda/2/m*(sum(sum(Theta1_trunc.^2))+sum(sum(Theta2_trunc.^2)));

%finding gradient for optimization
delta_3=a_3-Y; %output error
%delta_2=delta_3*Theta2_trunc.*sigmoidGradient(z_2); %hiddne layer
%delta2=delta_2(2:end);
%err_2=delta_3'*a_2;
%err_1=delta_2'*a_1;

delta_2=zeros([m hidden_layer_size+1]);
for i=1:m
    delta_2(i,:)=delta_3(i,:)*Theta2.*sigmoidGradient([1 z_2(i,:)]);
end
delta_2=delta_2(:,2:end);
err_2=zeros([num_labels hidden_layer_size+1]);
err_1=zeros([hidden_layer_size input_layer_size+1]) ;
for i=1:m
    err_2=err_2+delta_3(i,:)'*a_2(i,:);
    err_1=err_1+delta_2(i,:)'*a_1(i,:);
end
Theta1(:,1)=zeros(size(Theta1(:,1)));% remove bias factors
Theta2(:,1)=zeros(size(Theta2(:,1))); %remove bias factors
Theta1_grad=err_1/m;
Theta1_grad=Theta1_grad+lambda/m*Theta1;
Theta2_grad=err_2/m;
Theta2_grad=Theta2_grad+lambda/m*Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
