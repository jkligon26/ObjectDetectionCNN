%The folder path of the training images alter this if the
%folder path changes.
folderpath = 'D:\Documents\MATLAB\TrafficLightTrain\';

files = dir(fullfile(folderpath, '*.jpg'));
a = {'Filename','traffic'};
Traffics = cell2table(a);

%Opens all of the images one-by-one and allows for the 
%selection of ROIs around the target object.
for iterator = 1:length(files)
    filePath = fullfile(folderpath,files(iterator).name);
    I = imread(filePath);
    imshow(I);
    h = imrect;
    wait(h);
    ROIcap = getPosition(h);
    cellArray = {filePath, ROIcap};
    Traffics = [Traffics; cellArray];
end

close all;

Traffics.Properties.VariableNames{1} = 'Filename';
Traffics.Properties.VariableNames{2} = 'Traffic';
Traffics(1,:) = [];
