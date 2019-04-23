[X_trn_norm, y_trn, X_val_norm, y_val, X_tst_norm, y_tst] = ReadNormalizedOptdigitsDataset('optdigits_train.txt','optdigits_valid.txt','optdigits_test.txt');
Hs = [4,8,12,16,20,24];
training_error = zeros(length(Hs),1);
validation_error = zeros(length(Hs),1);
for i=1:length(Hs)
    H = Hs(i);
    [Y_pred,Z,W,V] = MLPTrain(X_trn_norm, y_trn, H);

    training_error(i) = CalculateErrorRate(Y_pred, y_trn);

    fprintf('Training set error rate when H=%d: %f\n', H, training_error(i));
    
    [Y,Z] = ForwardPropagation(X_val_norm, W, V);

    validation_error(i) = CalculateErrorRate(Y, y_val);
    
    fprintf('Validation set error rate when H=%d: %f\n', H, validation_error(i));
    
end

PlotTrainingValidationError(Hs, training_error, validation_error);


[m,i] = min(validation_error);
[Y_pred,Z,W,V] = MLPTrain(X_trn, y_trn, Hs(i));

[Y,Z] = ForwardPropagation(X_val_norm, W, V);

trainging_error(i) = CalculateErrorRate(Y, y_val);


fprintf('Test set error rate when H=%d: %f\n', Hs(idx), test_error);


[Y_trn,Z_trn,W,V] = MLPTrain(X_trn_norm, y_trn, 2);

[Y_val,Z_val] = ForwardPropagation(X_val_norm, W, V);
[Y_tst,Z_tst] = ForwardPropagation(X_tst_norm, W, V);

[m_trn, i_trn] = max(Y_trn,[],2);
[m_val, i_val] = max(Y_val,[],2);
[m_tst, i_tst] = max(Y_tst,[],2);
i_trn = i_trn - 1;
i_val = i_val - 1;
i_tst = i_tst - 1;
subplot(2,3,1);
PlotZ2DScatter(Z_trn,i_trn);
subplot(2,3,2);
PlotZ2DScatter(Z_val,i_val);
subplot(2,3,3);
PlotZ2DScatter(Z_tst,i_tst);

[Y_trn,Z_trn,W,V] = MLPTrain(X_trn_norm, y_trn, 3);

[Y_val,Z_val] = ForwardPropagation(X_val_norm, W, V);
[Y_tst,Z_tst] = ForwardPropagation(X_tst_norm, W, V);

[m_trn, i_trn] = max(Y_trn,[],2);
[m_val, i_val] = max(Y_val,[],2);
[m_tst, i_tst] = max(Y_tst,[],2);
i_trn = i_trn - 1;
i_val = i_val - 1;
i_tst = i_tst - 1;
subplot(2,3,4);
PlotZ3DScatter(Z_trn,i_trn);
subplot(2,3,5);
PlotZ3DScatter(Z_val,i_val);
subplot(2,3,6);
PlotZ3DScatter(Z_tst,i_tst);