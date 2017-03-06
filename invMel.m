function [ f ] = invMel( m )
%converts mels to frequency in Hz
    
    f = 700*(exp(m/1125) - 1);

end

