function [C,err_rate] = confmat(true_lb,est_lb)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARGUMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% true_lb : vector of n true labels (integers) attached to the data
% est_lb : vector of n estimated labels. The different labels in est_lb 
% must be elements of true_lb.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C : confusion matrix
% err_rate : total error rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind = unique(true_lb);
m = length(ind); % number of classes
C = accumarray([true_lb,est_lb],ones(length(true_lb),1));
C = C(ind,ind);
err_rate = sum(sum(C-diag(diag(C))))/sum(sum(C));
C = diag(1./sum(C,2))*C;
end

