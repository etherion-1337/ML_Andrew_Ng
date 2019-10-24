function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

summary_array = zeros(64,3);
row_pointer = 0;

for C_GridSearch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for sigma_GridSearch = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    ++row_pointer
    model = svmTrain(X,y,C_GridSearch, @(x1,x2) gaussianKernel(x1,x2,sigma_GridSearch));
    predictions = svmPredict(model, Xval);
    predictions_error = mean(double(predictions ~= yval));
    
    summary_array(row_pointer,:) = [C_GridSearch, sigma_GridSearch, predictions_error];
  end
end

% find min in col 3
[min_error, min_index] = min(summary_array(:,3));

C = summary_array(min_index,1);
sigma = summary_array(min_index,2);




% =========================================================================

end
