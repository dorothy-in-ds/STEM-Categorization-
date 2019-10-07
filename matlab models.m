clc; clear;
%% Define variables 
training = [1,1,1,0;
            1,0,1,0;
            1,0,1,1;
            1,1,0,1;
            0,1,1,1;
            1,1,0,0;
            0,1,1,0;
            0,0,0,1;
            0,0,0,0];
testing = [1,0,0,1;
           1,1,1,1;
           0,1,0,1;
           0,0,1,1;
           1,0,0,0;
           0,0,1,0;
           0,1,0,0]; 
train_n = ['4A' '7A' '15A' '13A' '5A' '12B' '2B' '14B' '10B']; 
test_n = ['1A' '6A' '9A' '11A' '3B' '8B' '16B']; 
all = vertcat(training, testing);
all_n = horzcat(train_n, test_n);
epsilon = [.1 .2 .3 .08]; 
params = 1 - epsilon; 

%% examplar model
% params = [.16 .16 .18 .14]; % based on Medin's context model
num_A_train = 5;
num_B_train = 4;
labels = horzcat(repmat('A',1,num_A_train), repmat('B',1,num_B_train));
A_probs = zeros(1,length(all));

for i = 1:length(all)
    s = all(i,:);
    s_n = all_n(i);
    distance = abs(training - s);
%     scores = prod((1.-distance) + distance.*params, 2);     % Medin's implementation
%     A_scores = sum(scores(labels == 'A'));
%     sum_scores = sum(scores)
%     A_probs(i) = A_scores/sum_scores;
    probs = prod((1.-distance).*params+distance.*epsilon, 2); % Bayesian approach
    scores = probs;
    A_scores = sum(scores(labels == 'A'));
    B_scores = sum(scores(labels == 'B'));

    A_probs(i) = A_scores/B_scores;
end 
 
examplar_res = max(A_probs, 1.-A_probs)';



%% prototype model
% params = [.38 .10 .40 .20];  % based on Medin's independent cues model 
labels = ['A' 'B'];
A_probs = zeros(1,length(all));
prototypes = [1 1 1 1;
              0 0 0 0];

for i = 1:length(all)
    s = all(i,:);
    s_n = all_n(i);
    distance = abs(prototypes-s);
%     scores = sum(((1.-distance)-distance).*params, 2);       % Medin's implementation
%     A_scores = sum(scores(labels == 'A'));
    probs = prod((1.-distance).*params+distance.*epsilon, 2); % Bayesian approach
    A_probs(i) = probs(labels == 'A')/probs(labels == 'B');
end 

prototype_res = max(A_probs, 1.-A_probs)';

%% Obtain rank order
prototype_sorted = sort(prototype_res, 'descend');
[~, prototype_rnk] = ismember(prototype_res, prototype_sorted);

examplar_sorted = sort(examplar_res, 'descend');
[~, examplar_rnk] = ismember(examplar_res, examplar_sorted);

prototype_rnk
examplar_rnk

% correlations between two models (after rank-ordered)
corrcoef(prototype_rnk, examplar_rnk)