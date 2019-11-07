classdef InputLayerFenchel
    properties
        InputSize
        Name = ' '
        State_in
        State_out
    end
    
    methods
        function lay = InputLayerFenchel(input_1,input_2)
            if nargin == 0
                error('Please specify the InputSize')
            elseif nargin == 1
                if isnumeric(input_1) && floor(input_1)  == input_1
                    lay.InputSize = input_1;
                else
                    error('Please specify an integer value for InputSize')
                end
            elseif nargin == 2
                
                if isnumeric(input_1) && floor(input_1)  == input_1
                    lay.InputSize = input_1;
                elseif ischar(input_1)
                    lay.Name = input_1;
                else
                    error('Please specify an integer value for InputSize and/or a string char for the name')
                end
                
                if ischar(input_1) && isnumeric(input_2) && floor(input_2)  == input_2
                    lay.InputSize = input_2;
                elseif ischar(input_2) && isnumeric(input_1)
                    lay.Name = input_2;
                else
                    error('Please specify an integer value for InputSize and/or a string char for the name')
                end   
                
            end
        end
    end
end