% Edward Gao, Don Dang
% ECEGR 4910
% 4 small projects

% Data source: Kaggle
% Link: https://www.kaggle.com/spscientist/students-performance-in-exams

clc; clear; close all; warning off;

fprintf("\n");
fprintf("========== DATA ==========\n");
fprintf("We obtained a student performance dataset from Kaggle and\n");
fprintf("we will use various classifiers on this dataset.\n");
fprintf("Reading data from csv file...\n");
FILENAME = "StudentsPerformance.csv";
data = readtable(FILENAME);

fprintf("Sneak peak of data:\n");
data(1:5, :)

fprintf("========== PURPOSE ==========\n");
fprintf("Based on a student's scores, we want to see if it is possible\n");
fprintf("to predict their parental level of education, lunch type, and\n");
fprintf("test preparation course since we postulate that these are\n");
fprintf("factors that influence their test scores, not the other way\n");
fprintf("around. By seeing which attribute's classification results\n");
fprintf("are closest to reality, we will assume that it implies that\n");
fprintf("particular attribute contributes most to a student's scores.\n");
fprintf("\n");

fprintf("========== FORMATTING INPUT DATA ==========\n");
fprintf("Format data so that there are only two inputs.\n");
fprintf("Averaging math, reading, and writing score as grades...\n");
grades = (data.mathScore + data.readingScore + data.writingScore) / 3;
fprintf("Checking unique values for parental level of education...\n");
unique(data.parentalLevelOfEducation)
fprintf("There are more than 2 classes!\n");
fprintf("Replacing values in PLE with 1 to 6...\n");
pleMap = containers.Map({'some high school', 'high school', ...
    'some college', 'associate''s degree', 'bachelor''s degree', ...
    'master''s degree'}, {1, 2, 3, 4, 5, 6});
oldPLE = data.parentalLevelOfEducation;
newPLE = zeros(size(oldPLE, 1), 1);
for i = 1:size(oldPLE)
    newPLE(i,1) = pleMap(char(oldPLE(i,1)));
end
fprintf("Creating feature vectors...\n");
dataX = [grades newPLE];
fprintf("\n");

fprintf("========== FORMATTING OUTPUT DATA ==========\n");
fprintf("Format data so that each attribute only has 2 classes.\n");
fprintf("Checking unique values for lunch type...\n");
unique(data.lunch)
fprintf("There are only 2 classes, OK!\n");
fprintf("Checking unique values for test preparation course...\n");
unique(data.testPreparationCourse)
fprintf("There are only 2 classes, OK!\n");

fprintf("Creating output vectors...\n");
dataY = [data.lunch data.testPreparationCourse];
fprintf("\n");

fprintf("========== STATISTICS BEFORE CLASSIFICATION ==========\n");
sizeY = size(dataY, 1);
fprintf("Lunch Type\n");
numStd = sum(ismember(dataY(:,1), {'standard'}));
fprintf("\tfree/reduced: %.2f%%\n", 100-numStd/sizeY*100);
fprintf("\tstandard: %.2f%%\n", numStd/sizeY*100);
fprintf("Test Preparation Course\n");
numNone = sum(ismember(dataY(:,2), {'none'}));
fprintf("\tnone: %.2f%%\n", numNone/sizeY*100);
fprintf("\tcompleted: %.2f%%\n", 100-numNone/sizeY*100);
fprintf("\n");

fprintf("========== CONVERT CLASSES TO NUMBERS ==========\n");
fprintf("Converting all classes to 1 and -1...\n");
newDataY = zeros(size(dataY));
for i = 1:size(dataY)
    curr = dataY(i,:);
    if ismember(curr(1), {'free/reduced'})
        newDataY(i,1) = -1;
    else
        newDataY(i,1) = 1;
    end
    if ismember(curr(2), {'none'})
        newDataY(i,2) = -1;
    else
        newDataY(i,2) = 1;
    end
end
dataY = newDataY;
fprintf("DONE!\n");
fprintf("\n");

fprintf("========== SETTING UP VARIOUS CLASSIFIERS ==========\n");
fprintf("Setting up general classifiers for each attribute...\n");
gcLunch = GC(dataX, dataY(:,1), 'Lunch Type', ...
             'Grades', 'Parental Level of Education');
gcTPC = GC(dataX, dataY(:,2), 'Test Preparation Course', ...
           'Grades', 'Parental Level of Education');
fprintf("Setting up various classifiers for lunch type...\n");
pnnLunch = PNN(gcLunch);
svmLunch = SVM(gcLunch);
fprintf("Setting up various classifiers for TPC...\n");
pnnTPC = PNN(gcTPC);
svmTPC = SVM(gcTPC);
fprintf("\n");

fprintf("========== LUNCH TYPE ==========\n");
pnnLunch.train();
svmLunch.train();
fprintf("TRAIN SET ERROR\n");
[t, c, i] = pnnLunch.trainError();
fprintf("PNN: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
[t, c, i] = svmLunch.trainError();
fprintf("SVM: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
fprintf("TEST SET ERROR\n");
[t, c, i] = pnnLunch.testError();
fprintf("PNN: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
[t, c, i] = svmLunch.testError();
fprintf("SVM: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
fprintf("\n");

fprintf("========== TEST PREPARATION COURSE ==========\n");
pnnTPC.train();
svmTPC.train();
fprintf("TRAIN SET ERROR\n");
[t, c, i] = pnnTPC.trainError();
fprintf("PNN: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
[t, c, i] = svmTPC.trainError();
fprintf("SVM: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
fprintf("TEST SET ERROR\n");
[t, c, i] = pnnTPC.testError();
fprintf("PNN: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
[t, c, i] = svmTPC.testError();
fprintf("SVM: %d total, %d correct (%.2f%%), %d incorrect (%.2f%%)\n", ...
        t, c, c/t*100, i, i/t*100);
fprintf("\n");