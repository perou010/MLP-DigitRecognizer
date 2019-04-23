% The parameters received are:
% - X_trn (N x D): Training datapoints matrix, where N is the 
% number of training data points, and D is the number of features
% - y_trn (N x 1): Vector that should contain the labels of the 
% training datapoints
% - H (1 x 1): Number of hidden units
%
% The function should return:
% - Y_pred (N x K): Output from the last Forward Propagation
% - Z (N x H+1): Matrix that contains the output from the hidden units,
% including the bias unit z0=+1
% - W (D+1 x H): Weights learned between each input unit and hidden unit
% - V (H+1 x K): Weights learned between each hidden unit and output unit
%
function [Y_pred,Z,W,V] = MLPTrain(X_trn, y_trn, H)
    
    K = 10;
    D = size(X_trn,2);
    maxiter = 1000;
    eta = 0.0005;

    rng(1); % For reproducibility
    W = -0.01 + (0.01+0.01)*rand(D+1,H);
    rng(2); % For reproducibility
    V = -0.01 + (0.01+0.01)*rand(H+1,K);

    [Y,Z] = ForwardPropagation(X_trn, W, V);
    E = ErrorFunction(y_trn, Y);
    %%%% 

    dV_old = 0;
    dW_old= 0;
    ErrorArray = [0 0 0 0 0 0];
    a = .0002;
    b = .0004;
    for iter=1:maxiter
        % Find updates dW and dV, using BackwardPropagation, and update W 
        % and V
        if (mean(ErrorArray) > ErrorArray(mod(iter,6)+1))
            %eta = eta + a;
        else
            %iter
            %eta  = eta - b * eta;
        end
        
        alpha = 0.5;
        [dW_new, dV_new] = BackwardPropagation(X_trn, y_trn, Y, Z, V, eta);
        V = V + dV_new + alpha * dV_old;
        W = W + dW_new + alpha * dW_old;
        dV_old = dV_new;
        dW_old = dW_new;

        [Y,Z] = ForwardPropagation(X_trn, W, V);
        E_new = ErrorFunction(y_trn, Y);

        if abs(E < E_new) <0.2
            break;
        end
        %E - E_new
        E = E_new;
        %ErrorArray(mod(iter+1, 6)+1) = E;
    end
    Y_pred = Y;
end

