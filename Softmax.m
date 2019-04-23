function [softmaxvalues] = Softmax(Y_pred, base)
    exps = exp(Y_pred*base);
    expsum = sum(exps);
    softmaxvalues = exps/expsum;
end

