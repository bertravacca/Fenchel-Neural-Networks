classdef OptionsFenchel
    properties
        update_order='backward'
        bcd_solver='cvx'
        method='bcd'
        FenchelBackPenalization=10^-2
        learning_rate_method='constant'
        learning_rate
        history_f_val_fenchel=0
    end
    
    methods
        function  options = OptionsFenchel(varargin)
            possProperties = {'update_order','bcd_solver','method','FenchelBackPenalization','history_f_val_fenchel','learning_rate','learning_rate_method'};
            k=1;
            while k<nargin
                switch varargin{k}
                    case possProperties
                        options.(varargin{k})=varargin{k+1};
                        k=k+2;
                    otherwise
                        possibilities=strjoin(possProperties,', ');
                        if ischar(varargin{k})
                            error(['Unknown property for OptionsFenchel class:', varargin{k},', possible specifications include: ',possibilities])
                        else
                            error(['Please check that your property specifictions are correct. Possible specifications include: ', possibilities])
                        end
                end
            end
            % check that the specs make sense
            if isnumeric(options.FenchelBackPenalization)==0
                error('FenchelBackPenalization must be a double')
            end
            
            if strcmp(options.method,'bcd')==0
                possMethods={'gradient','bcd','accelerated_gradient','bcd_gradient'};
                possibilities=strjoin(possMethods,', ');
                method_implemented='no';
                for k=1:length(possMethods)
                    if strcmp(possMethods{k},options.method)
                        method_implemented='yes';
                    end
                end
                if strcmp(method_implemented,'no')
                    error(['The specified method: ', options.method, 'does not exist, possibilities include: ', possibilities])
                end
            end
            
            if options.history_f_val_fenchel~=0 && options.history_f_val_fenchel~=1
                error('the OptionsFenchel history_f_val_fenchel property is either 0 [do not keep history] or 1 [keep history]')
            end
            
            if isempty(options.learning_rate)==0
                if isnumeric(options.learning_rate)==0|| max(size(options.learning_rate))~=1
                    error('The step size value must be a 1x1 double')
                end
               options.learning_rate_method='constant';
            end

            if strcmp(options.learning_rate_method,'constant')==0
                possMethods={'backtracking'};
                possibilities=strjoin(possMethods,', ');
                method_implemented='no';
                for k=1:length(possMethods)
                    if strcmp(possMethods{k},options.learning_rate_method)
                        method_implemented='yes';
                    end
                end
                if strcmp(method_implemented,'no')
                    error(['The specified step size method: ', options.method, 'does not exist, possibilities include: ', possibilities])
                end
            end
            

        end
    end
end
