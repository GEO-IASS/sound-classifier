function [ m ] = mel( f )
%converts frequency to mel scale

    m = 1125 * log(1+f/700);


end

