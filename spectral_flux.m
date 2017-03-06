clear

%global vars
fs = 44100;

%windowing
winLen = 512;
overlap = 0.5;
hannWin = hann(winLen);

%calculating the filterbank for MFCC's extraction
filterNo = 26;
filterL = winLen/2 + 1;
filterbank = zeros([filterL, filterNo]);
loFreq = 80;
hiFreq = fs;

melPoints = zeros([filterNo + 2, 1]);
freqPoints = zeros([filterNo + 2, 1]);
binPoints = zeros([filterNo + 2, 1]);
loMel = mel(loFreq);
hiMel = mel(hiFreq);
for j = 1:filterNo + 2
    melPoints(j) = loMel + (hiMel-loMel)*((j-1)/(filterNo+1));
    freqPoints(j) = invMel(melPoints(j));
    binPoints(j) = floor(freqPoints(j)/(fs/(filterL-1)));
end

for i = 1:filterNo
    beginBin = binPoints(i);
    peakBin = binPoints(i+1);
    endBin = binPoints(i+2);
    
    for j = beginBin+1:endBin-1
        if j <= peakBin
            value = (j - beginBin) / (peakBin - beginBin);
        elseif j > peakBin
            value = (endBin - j) / (endBin - peakBin);
        end
        filterbank(uint32(j), uint32(i)) = value;
    end
end

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
    
    %fileID = fopen(datasetFiles{classID}, 'w');
    files = dir(soundDirs{classID});
    fileIndex = find(~[files.isdir]);
    
    for fi = fileIndex
        
        filename = strcat(soundDirs{classID}, files(fi).name)

        %filename = 'C:\Users\Dominik\Desktop\ISEL\SaU\Final_Project\sounds\doors\Door_Close-SoundBible.wav';
        [soundData, fs] = audioread(filename);
        
        %peak normalization
        peakval = max(soundData);
        if(peakval < 1.0)
            soundData = soundData * (1.0/peakval);
        end

        %extending the sound to fit the winLen
        if mod(length(soundData), winLen) ~= 0
            extraZeros = zeros([winLen - mod(length(soundData), winLen), 1]);
            soundData = vertcat(soundData, extraZeros);
        end


        step = winLen / 2;
        totalWindowNo = (length(soundData)-winLen)/step;
        spectralFlux = zeros([totalWindowNo, 1]);
        maxSpectralFlux = 0.02;
        onsetVector = double.empty;
        mfcc = zeros([filterNo, totalWindowNo]);
        mfccRange = 2:13;
        overload = 0;
        for i = 1:step:length(soundData)-winLen

            winNo = (i-1)/step+1;    
            window = soundData(i:i+winLen-1).*hannWin;

            windowFft = fft(window);    
            windowFft = abs(windowFft/winLen);
            windowFft = windowFft(1 : winLen/2 + 1);

            if i>1
                %calculating spectral flux
                for j = 1:length(windowFft)
                    spectralFlux(winNo) = spectralFlux(winNo) + (windowFft(j)-prevWinFft(j))^2;
                    maxSpectralFlux = max(spectralFlux(winNo), maxSpectralFlux);
                end
            end

            %calculating MFCC's
            filterEnergy = zeros([filterNo, 1]);
            for j = 1:filterNo
                energy = 0;
                for k = 1:filterL
                    energy = energy + windowFft(k)*filterbank(k, j);
                end
                filterEnergy(j) = log(energy);
            end
            mfcc(:, winNo+1) = dct(filterEnergy);


            %thresholding
            threshold = maxSpectralFlux*0.5;
            if spectralFlux(winNo) > threshold && overload == 0
                onsetVector = [onsetVector, winNo];
                overload = overload + (threshold^2)*10;
            end    
            overload = overload + spectralFlux(winNo)^2 - threshold^2;
            overload = max(0, overload);


            if(overload > 0)
                %bar(mfcc(:, winNo+1), 'r')
                for j = mfccRange
                    %fprintf(fileID,'%5.2f,',mfcc(j, winNo+1));
                end
                %fprintf(fileID,'\n');
            else
                %bar(mfcc(:, winNo+1), 'b')
            end    
            ylim([-10, 10]);
            xlim([1.5, 13.5]);
            %sound(window);
            %drawnow

            prevWinFft = windowFft;

        end

        plot(spectralFlux);
        ylim([0, 0.1]);
        drawnow
        
    end
end
