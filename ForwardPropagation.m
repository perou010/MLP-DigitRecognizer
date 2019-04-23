% The parameters received are:
% - X (N x D): Training datapoints matrix, where N is the 
% number of training data points, and D is the number of features
% - W (D+1 x H): Weights between each input unit and hidden unit
% - V (H+1 x K): Weights between each hidden unit and output unit
%
% The function should return:
% - Y (N x K): Output of each output unit
% - Z (N x H+1): Output of each hidden units, including the bias unit z0=+1
%
function [Y,Z] = ForwardPropagation(X, W, V)
zero = zeros(size(X,1),1);
one = zero + 1;
X = [X one];
Z = Sigmoid(X*W);
Z = [Z one];
Y = Z * V;
end

