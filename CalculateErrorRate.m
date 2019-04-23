% Calculates the error rate (percentage of wrongly classified samples)
%
% The parameters received are:
% - Y_pred (N x K): Output of each output unit
% - y_label (N x 1): True labels of each data point, where N is the number
% of data points
%
% The function should return:
% - error_rate (1 x 1): The error rate (between 0 and 1)
%
function error_rate = CalculateErrorRate(Y_pred,y_label)
errors = 0;
for x = 1:size(y_label,1)
    softmaxVals = Softmax(Y_pred(x,:),1);
    [mx, indx] = max(softmaxVals);
    if y_label(x) ~= indx-1
        errors = errors + 1;
    end
end
error_rate = errors / size(y_label,1);
end