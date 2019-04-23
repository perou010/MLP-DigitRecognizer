function [X_trn_norm, y_trn, X_val_norm, y_val, X_tst_norm, y_tst] = ReadNormalizedOptdigitsDataset(training_filename, validation_filename, test_filename)
Training = importdata(training_filename);
Validation = importdata(validation_filename);
Testing = importdata(test_filename);
X_trn = Training(1:end,1:end-1);
y_trn = Training(1:end,end);
X_val = Validation(1:end,1:end-1);
y_val = Validation(1:end,end);
X_tst = Testing(1:end,1:end-1);
y_tst = Testing(1:end,end);
mu = mean(X_trn);
sigma = std(X_trn);
sigma(sigma == 0) = 1;
X_trn_norm = (X_trn-mu)./sigma;
X_tst_norm = (X_tst-mu)./sigma;
X_val_norm = (X_val-mu)./sigma;
end

