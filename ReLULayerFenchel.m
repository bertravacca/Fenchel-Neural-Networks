classdef ReLULayerFenchel
    properties
        Name = ' '
        Activation='ReLU'
        OutputSize=0
        InputSize
        L1_weight_regularization_param=0
        L2_weight_regularization_param=0
        Weight_init
        Weight
        Weight_prev
        Bias_init
        Bias
        Bias_prev
        State_in
        State_in_prev
        State_out
        State_out_prev
        learning_rate
        Gradients
    end
    
    methods
        function lay = ReLULayerFenchel(varargin)
            possProperties = {'Name','OutputSize','L1_weight_regularization_param','L2_weight_regularization_param','Weight_init','Bias_init'};
            k=1;
            while k<nargin
                switch varargin{k}
                    case possProperties
                        lay.(varargin{k})=varargin{k+1};
                        k=k+2;
                    otherwise
                        possibilities=strjoin(possProperties,', ');
                        if ischar(varargin{k})
                            error(['Unknown property for ReLULayerFenchel class:', varargin{k},', possible specifications include: ',possibilities])
                        else
                            error(['Please check that your property specifictions are correct, possible specifications include: ',possibilities])
                        end
                end
            end
            % check that the specs make sense
            if lay.OutputSize==0
                error('Please specify the OutputSize')
            end
            if isnumeric(lay.OutputSize)==0 || length(lay.OutputSize)>1 ||  floor(lay.OutputSize)~=lay.OutputSize
                error('The OutputSize must be a 1x1 integer value')
            end
            if isnumeric(lay.L1_weight_regularization_param)==0
                error('The OutputSize must be a 1x1 double')
            end
            
            if isnumeric(lay.Weight_init)==0
                error('The weight Weight_init must be numeric')
            elseif isempty(lay.Weight_init)==0
                [m,~]=size(lay.Weight_init);
                if m~=lay.OutputSize
                    error('The number of rows for the weight Weight_init should be the same as the Outputsize')
                end
            end
            
            if isnumeric(lay.Bias_init)==0
                error('The bias Bias_init must be numeric')
            elseif isempty(lay.Bias_init)==0
                [m,c]=size(lay.Bias_init);
                if m~=lay.OutputSize
                    error('The bias should be a column vector of the same size as the Outputsize')
                end
                if c~=1
                    error('The bias must be a column vector')
                end
            end
        end
    end
end