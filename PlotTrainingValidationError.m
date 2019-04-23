% The parameters received are:
% - Hs (#Hs x 1): Vector with all the H options
% - training_error (#Hs x 1): Training set error rate for each H
% - validation_error (#Hs x 1): Validation set error rate for each H
%
function PlotTrainingValidationError(Hs,training_error, validation_error)
hold on
for i = 1:size(Hs,2)
    scatter(Hs(i),training_error(i),[],'r');
    scatter(Hs(i),validation_error(i),[],'b');
end
end
