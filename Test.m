clc; clear; close all;

X = [randn(30,2)+2; randn(30,2)];
Y = [ones(30,1); -ones(30,1)];
gc = GC(X, Y, 'Random Set');

% svm = SVM(gc);
% svm.train();
% [t, c, i] = svm.trainError();
% fprintf("total: %d, correct: %d, incorrect: %d\n", t, c, i);
% [t, c, i] = svm.testError();
% fprintf("total: %d, correct: %d, incorrect: %d\n", t, c, i);

pnn = PNN(gc);
pnn.train();
[t, c, i] = pnn.trainError();
fprintf("total: %d, correct: %d, incorrect: %d\n", t, c, i);
[t, c, i] = pnn.testError();
fprintf("total: %d, correct: %d, incorrect: %d\n", t, c, i);