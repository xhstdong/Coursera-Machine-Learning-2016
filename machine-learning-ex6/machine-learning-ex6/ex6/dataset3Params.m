function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

error=1000000;
parameter=[0.01,0.03,0.1,0.3,1,3,10,30];
%find optimal sigma
for i=1:size(parameter,2)
    test_sigma=parameter(i);
    for j=1:size(parameter,2)
        test_C=parameter(j);
        model= svmTrain(X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_sigma)); %training set
        predictions=svmPredict(model, Xval); %validation set
        test_error=mean(double(predictions ~=yval));
        if test_error<error %update value if new parameters work better
            error=test_error;
            C=test_C;
            sigma=test_sigma;
        end
    end
end
% =========================================================================

end
