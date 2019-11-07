classdef OutputLayerFenchel
    properties
        Name = ' '
        Type='regression'
        OutputSize=0
        InputSize
        L1_weight_regularization_param=0
        L2_weight_regularization_param=0
        Weight_init
        Weight
        Weight_prev
        Weight_tmp
        Bias_init
        Bias
        Bias_prev
        Bias_tmp
        State_in
        State_in_prev
        State_in_tmp
        State_out
        State_out_prev
        State_out_tmp
        learning_rate
        Gradients
    end
    
    methods
        function lay = OutputLayerFenchel(varargin)
            possProperties = {'Name','OutputSize','L1_weight_regularization_param','L2_weight_regularization_param','Weight','Weight_init','Bias','Bias_init'};
            k=1;
            while k<nargin
                switch varargin{k}
                    case possProperties
                        lay.(varargin{k})=varargin{k+1};
                        k=k+2;
                    otherwise
                        possibilities=strjoin(possProperties,', ');
                        if ischar(varargin{k})
                            error(['Unknown property for OutputLayerFenchel class:', varargin{k},', possible specifications include: ', possibilities])
                        else
                            error(['Please check that your property specifictions are correct. Possible specifications include: ', possibilities])
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
        end
    end
end