classdef LA
    % Linear Algebra: A class implementing various functions and
    % linear algebraic utilities.
    %    
    % (c) Panos Achlioptas 2016    http://www.stanford.edu/~optas

    methods(Static)
        
        function y = outersum(a, b)
            a = a(:);
            b = b(:)';
            y = repmat(a, size(b)) + repmat(b, size(a));
        end        
    end
    
end
