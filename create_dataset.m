clear

%preparing filenames and directories
datasetFolder = 'dataset/';
datasetNames = {'car', 'doors', 'glass', 'shots', 'thunder'};
datasetFormat = '.csv';
datasetFiles = char.empty;
soundFolder = 'sounds/';
soundDirs = char.empty;
for datasetName = datasetNames
    datasetFiles = [datasetFiles, strcat(datasetFolder, datasetName, datasetFormat)];
    soundDirs = [soundDirs, strcat(soundFolder, datasetName, '/')];
end

for classID = 1:length(datasetNames)
    
    fileID = fopen(datasetFiles{classID}, 'w');
    files = dir(soundDirs{classID});
    fileIndex = find(~[files.isdir]);
    
    for fi = fileIndex
        
        filename = strcat(soundDirs{classID}, files(fi).name)

        [soundData, fs] = audioread(filename);
        
        soundEvents = mfcc(soundData, fs);
        
        for ei = 1:length(soundEvents)
            fprintf(fileID,'%s #%d\n',filename,ei);
            [cols, rows] = size(soundEvents{ei});
            event = soundEvents{ei};
            for i = 1:rows
                for j = 1:cols
                    fprintf(fileID,'%5.2f,',event(j, i));
                end
                fprintf(fileID,'\n');
            end
        end
        
    end
end
