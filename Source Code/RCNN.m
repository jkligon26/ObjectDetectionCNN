cifar10Data = tempdir;

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';

%This portion downloads, initializes, and formats the CIFAR data for
%training.
helperCIFAR10Data.download(url, cifar10Data);

[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load(cifar10Data);
numImageCategories = 10;

[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];

%Initializing filter size and number of filters for CNN.
filterSize = [5 5];
numFilters = 40;

%Create neural network architecture.
layers = [
    imageInputLayer(imageSize);

    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride', 2)

    convolution2dLayer(filterSize, numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)

    convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)
  
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numImageCategories)
    softmaxLayer
    classificationLayer
    ];

%Randomizing the intial weights of the CNN.
layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

%Setting options and some hyperparameters of the CNN.
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

%Training our regular CNN with CIFAR.
cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);

YTest = classify(cifar10Net, testImages);

%Output accuracy of network performance for hyperparameter testing and to
%observe the performance.
accuracy = sum(YTest == testLabels)/numel(testLabels)

data = load('Traffics.mat', 'Traffics');
traffic = data.Traffics;

% Setting training options
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);

% Train the R-CNN using pre-trained CNN.
rcnn = trainRCNNObjectDetector(traffic, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])

testingPath = 'D:\Documents\MATLAB\TrafficLightTest\';
testImages = dir(fullfile(testingPath, '*.jpg'));

%Loads every image in the test image directory for testing in the R-CNN.
for iterator = 1:length(testImages)
    filePath = fullfile(testingPath,testImages(iterator).name);
    testImage = imread(filePath);
    
    %Calculates the detection scores and draws a rectangular ROI over the 
    %detecter object.
    [bboxes, score, label] = detect(rcnn, testImage, 'MiniBatchSize', 128);
    [score, idx] = max(score);
    bbox = bboxes(idx, :);
    annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
    outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);
    
    imshow(outputImage);
    
    w = waitforbuttonpress; 
end

close all;