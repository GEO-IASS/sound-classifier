function [ X ] = mfcc( soundData, fs )
%mfcc, computes and returns Mel Frequency Cepstral Coefficients of
%soundData in 512-sample windows

%%%%%%%%%%
%constants
%windowing
winLen = 512;
overlap = 0.5;
hannWin = hann(winLen);
%thresholding
minimalMaxSpectralFlux = 0.02;
thresholdingLevel = 0.5;
eventNo = 0;
%mfcc
mfccRange = 2:13;
%%%%%%%%%%%%%%%%%

%%%%%%
%flags
displayOn = false;
inEvent = false;
%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


step = winLen * overlap;
totalWindowNo = (length(soundData)-winLen)/step;
spectralFlux = zeros([totalWindowNo, 1]);
maxSpectralFlux = minimalMaxSpectralFlux;
onsetVector = int32.empty;
offsetVector = int32.empty;
mfccArray = zeros([filterNo, totalWindowNo]);
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

    %%%%%%%%%%%%%
    %thresholding
    threshold = maxSpectralFlux*thresholdingLevel;
    %onset
    if spectralFlux(winNo) > threshold && inEvent == false
        onsetVector = [onsetVector, winNo];
        overload = overload + (threshold^2)*10;
        inEvent = true;
        eventNo = eventNo + 1;
    end    
    %offset
    if inEvent && overload == 0
        offsetVector = [offsetVector, winNo];
        inEvent = false;
    end
    overload = overload + spectralFlux(winNo)^2 - threshold^2;
    overload = max(0, overload);


    if(inEvent)
        %calculating MFCC's
        filterEnergy = zeros([filterNo, 1]);
        for j = 1:filterNo
            energy = 0;
            for k = 1:filterL
                energy = energy + windowFft(k)*filterbank(k, j);
            end
            filterEnergy(j) = log(energy);
        end
        mfccArray(:, winNo) = dct(filterEnergy);

        if displayOn 
            bar(mfccArray(:, winNo), 'r');
        end
    else        
        if displayOn 
            bar(mfccArray(:, winNo), 'b'); 
        end
    end    
    if displayOn
        ylim([-10, 10]);
        xlim([1.5, 13.5]);
        sound(window);
        drawnow
    end

    prevWinFft = windowFft;

end

if inEvent
    offsetVector = [offsetVector, winNo];
end

%constructing output
X = cell([length(onsetVector), 1]);

for i = 1:length(onsetVector)
    b = onsetVector(i);
    e = offsetVector(i);
    
    X{i} = mfccArray(mfccRange, b:e-1);
end

plot(spectralFlux);
ylim([0, 0.1]);
drawnow
    

end

