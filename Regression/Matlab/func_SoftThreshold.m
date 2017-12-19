%% function to conduct soft thresholding
%
%   min_{y} ||x-y||_2 + lambda*||y||_1
%
%
function y = func_SoftThreshold(x,lambda)

if ~exist('x','var')
    x = rand(100,1)-0.5;
end

if ~exist('lambda','var')
    lambda = 0.1;
end

y_pos_tentative = x-0.5*lambda;
y_neg_tentative = x+0.5*lambda;

y = y_pos_tentative.*(y_pos_tentative>0) + y_neg_tentative.*(y_neg_tentative<0);


