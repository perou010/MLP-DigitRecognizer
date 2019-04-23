% Implements the MLP Backward Propagation step
%
% The parameters received are:
% - X (N x D): Training datapoints matrix, where N is the 
% number of training data points, and D is the number of features
% - y_label (N x 1): True labels of each data point
% - Y_pred (N x K): Output of each output unit
% - Z (N x H+1): Matrix that contains the output from the hidden units,
% including the bias unit z0=+1
% - V (H+1 x K): Weights between each hidden unit and output unit
% - eta (1 x 1): The learning rate
%
% The function should return:
% - dW (D+1 x H): Updates for the weights in W
% - dV (H+1 x K): Updates for the weights in V
%
function [dW, dV] = BackwardPropagation(X, y_label, Y_pred, Z, V, eta)
dV = zeros(size(V));
for x = 1:size(X,1)
    for h = 1:size(Z,2)
    Y_softmax = Softmax(Y_pred(x,:), 1);
    indicator = zeros(1, 10);
    indicator(y_label(x)+1) = 1;
    dV(h,:) = dV(h,:) + (indicator - Y_softmax)*Z(x,h);
    end
end
dV = dV * eta;
dW = zeros(size(X,2)+1, size(V,1)-1);
for j = 1:size(X,2)+1
    for x = 1:size(X,1)
        for h = 1:size(Z,2)
            Y_softmax = Softmax(Y_pred(x,:), 1);
            indicator = zeros(1, 10);
            indicator(y_label(x)+1) = 1;      
            if h ~= size(Z,2)
                if j ~= size(X,2)+1
                    dW(j,h) = dW(j,h) + (dot((indicator - Y_softmax),V(h,:))) * (Z(x,h)) * (1 - Z(x,h)) * (X(x,j));   
                else
                    dW(j,h) = dW(j,h) + (dot((indicator - Y_softmax),V(h,:))) * (Z(x,h)) * (1 - Z(x,h));  
                end
            end

        end
    end
end
dW = dW*eta;
end

