% Calculates the error function according to the current predictions
%
% The parameters received are:
% - y_label (N x 1): True labels of each data point, where N is the number
% of data points
% - Y_pred (N x K): Output of each output unit, where K=10 (0 to 9)
%
% The function should return:
% - E (1 x 1): The value of the error function
%
function E = ErrorFunction(y_label,Y_pred)
E = 0; 
for x = 1:size(y_label,1)
    softmaxVals = Softmax(Y_pred(x,:), 1);
    indicator = zeros(1, 10);
    indicator(y_label(x)+1) = 1; 
    lgvals = log(softmaxVals);
    E = E + dot(indicator, lgvals);
end
E = -E;
end

