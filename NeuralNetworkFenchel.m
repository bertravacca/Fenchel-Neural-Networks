classdef NeuralNetworkFenchel
    properties
        input_data
        output_data
        layers
        options
        num_train
        num_features
        num_outputs
        num_layers
        rmse
        f_val_fenchel_history
        loss
    end
    
    %% METHODS
    methods
        % Constructor
        function net=NeuralNetworkFenchel(X,Y,layers,options)
            net.input_data=X;
            net.output_data=Y;
            net.layers=layers;
            if nargin>3
                net.options=options;
            else
                net.options=OptionsFenchel;
            end
            % check data
            if isnumeric(X)==0 || isnumeric(Y)==0
                error('input and output data must be a numeric array')
            end
            
            [num_row_1,num_col_1]=size(X); [num_row_2,num_col_2]=size(Y);
            if num_col_1~=num_col_2
                error('innput and output data X, Y must be provided as matrices with number of columns corresponding to the number a datapoints in the training set')
            else
                net.num_train=num_col_1;
                net.num_features=num_row_1;
                net.num_outputs=num_row_2;
            end
            
            % check layers class
            num_layers=length(layers);
            if isa(layers{1},'InputLayerFenchel')==0
                error('The input layer should be of the class InputLayerFenchel')
            end
            for k=2:num_layers-1
                if isa(layers{k},'ReLULayerFenchel')==0
                    error('The hidden layers should be of the class ReLULayerFenchel')
                end
            end
            if isa(layers{num_layers},'OutputLayerFenchel')==0
                error('The output layer should be of the class OutputLayerFenchel')
            end
            
            net.num_layers=num_layers-1;
            % initialize weights
            for k=2:net.num_layers+1
                if isempty(net.layers{k}.Bias_init)
                    net.layers{k}.Bias=rand(net.layers{k}.OutputSize,1)-0.5;
                else
                    net.layers{k}.Bias=net.layers{k}.Bias_init;
                end
                if k==2
                    if isempty(net.layers{k}.Weight_init)
                        net.layers{k}.Weight=rand(net.layers{k}.OutputSize,net.num_features)-0.5;
                    else
                        net.layers{k}.Weight=net.layers{k}.Weight_init;
                    end
                else
                    if isempty (net.layers{k}.Weight_init)
                        net.layers{k}.Weight=rand(net.layers{k}.OutputSize,net.layers{k-1}.OutputSize)-0.5;
                    else
                        net.layers{k}.Weight=net.layers{k}.Weight_init;
                    end
                end
            end
            disp('-------------------------------------------------------------------------------')
            disp('-------------------Fenchel neural net constructed-------------------')
        end
        
        % Train Fenchel Neural Net
        function net=trainNeuralNetwork(net,num_iterations)
            L=net.num_layers;
            net.rmse=NaN*zeros(num_iterations+1,1);
            net.rmse(1)=net.compute_rmse_feedforward(net.input_data,net.output_data);
            disp('-------------------------------------------------------------------------------')
            disp(['Training started, initial RMSE=',num2str(net.rmse(1))])
            net=net.feedforward_neural_network;
            
            %% BCD method with CVX
            if strcmp(net.options.method,'bcd') && strcmp(net.layers{L+1}.Type,'regression')&& strcmp(net.options.update_order,'backward')&&strcmp(net.options.bcd_solver,'cvx')
                if net.options.history_f_val_fenchel==1
                    net.f_val_fenchel_history=NaN*zeros(num_iterations,1);
                end
                for iter=1:num_iterations
                    for k=flip(2:L+1)
                        if k==L+1
                            [net.layers{k}.Weight,net.layers{k}.Bias]=net.bcd_unconstrained_output_weight_regression;
                            net.layers{k}.State_out=net.layers{k}.Weight*net.layers{k}.State_in+net.layers{k}.Bias;
                        elseif k==L
                            [net.layers{k}.State_out]=net.bcd_fenchel_output_state_regression;
                            net.layers{k+1}.State_in=net.layers{k}.State_out;
                            [net.layers{k}.Weight,net.layers{k}.Bias]=net.bcd_fenchel_unconstrained_hidden_weight(k);
                        else
                            [net.layers{k}.Weight,net.layers{k}.Bias]=net.bcd_fenchel_unconstrained_hidden_weight(k);
                            [net.layers{k}.State_out]=net.bcd_fenchel_hidden_state_out(k);
                            net.layers{k+1}.State_in=net.layers{k}.State_out;
                        end
                    end
                    net.rmse(iter+1)=net.compute_rmse_feedforward(net.input_data,net.output_data);
                    disp('-------------------------------------------------------------------------------')
                    disp(['BCD iteration #',num2str(iter),' finished, RMSE=',num2str(net.rmse(iter+1))])
                    if net.options.history_f_val_fenchel==1
                        net.f_val_fenchel_history(iter)=net.f_val_fenchel;
                    end
                end
            end
               
            if strcmp(net.options.method,'bcd_gradient') && strcmp(net.layers{L+1}.Type,'regression')&& strcmp(net.options.update_order,'backward')
                disp('hellllooooo? in here')
                for iter=1:num_iterations
                    net=net.fenchel_full_bcd_backward_one_step_gradient;
                    net.rmse(iter+1)=net.compute_rmse_feedforward(net.input_data,net.output_data);
                    if floor(10*iter/num_iterations)==10*iter/num_iterations
                        disp('-------------------------------------------------------------------------------')
                        disp(['BCD iteration #',num2str(iter),' finished, RMSE=',num2str(net.rmse(iter+1))])
                    end
                    if net.options.history_f_val_fenchel==1
                        net.f_val_fenchel_history(iter)=net.f_val_fenchel;
                    end
                end
                
            end
            
            %% first order methods
            if strcmp(net.options.method,'gradient') || strcmp(net.options.method,'accelerated_gradient')&& strcmp(net.layers{L+1}.Type,'regression')
                net=net.feedforward_neural_network;
                if net.options.history_f_val_fenchel==1
                    net.f_val_fenchel_history=NaN*zeros(num_iterations,1);
                end
                
                for iter=1:num_iterations
                    % make a copy of X and W
                    net=net.make_copies_prev;
                    
                    % constant learning rate
                    if  strcmp(net.options.learning_rate_method,'constant') && strcmp(net.options.method,'gradient')
                        rate=net.options.learning_rate;
                        net=net.fenchel_full_gradient_update(rate);
                    end
                    
                    % accelerated method
                    if strcmp(net.options.method,'accelerated_gradient')
                        rate=net.options.learning_rate;
                        net=net.fenchel_full_accelerated_gradient_update(rate,iter);
                    end
                    
                    % backtracking line search
                    if strcmp(net.options.learning_rate_method,'backtracking')
                        net=net.fenchel_full_gradient;
                        if iter==1
                            initial_rate=1/net.num_train;
                        end
                        beta=0.5;
                        diff_f_val=NaN*zeros(10^3,1);
                        num=10;
                        k_target=ceil(num/2);
                        k=1;
                        while  k<10^3 && (k<=num || diff_f_val(k)>0)
                            rate=beta^k*initial_rate;
                            net=net.paste_copies_prev;
                            net=net.fenchel_full_gradient_update(rate);
                            diff_f_val(k)=net.f_val_fenchel-net.f_val_fenchel_prev;
                            k=k+1;
                        end

                        [diff_f,k_optimal]=min(diff_f_val);
                        rate=(beta^(k_optimal))*initial_rate;
                        initial_rate=(1/beta)^(k_target-k_optimal)*initial_rate;
                        net=net.paste_copies_prev;
                        net=net.fenchel_full_gradient_update(rate);
                        
                        %disp(['min f diff: ', num2str(diff_f)])
                        %disp(['actual f diff: ', num2str(net.f_val_fenchel-net.f_val_fenchel_prev)])
                        %disp(['k opt',num2str(k_optimal)])
                        %disp(['step size: ',num2str(rate)])
                    end
                    
                    net.rmse(iter+1)=net.compute_rmse_feedforward(net.input_data,net.output_data);
                    if floor(10*iter/num_iterations)==10*iter/num_iterations
                        disp('-------------------------------------------------------------------------------')
                        disp(['Gradient iteration #',num2str(iter),' finished, RMSE=',num2str(net.rmse(iter+1))])
                    end
                    if net.options.history_f_val_fenchel==1
                        net.f_val_fenchel_history(iter)=net.f_val_fenchel;
                    end
                end
            end
        end
        
        %% Feedforward methods
        % feedforward that instantiates the layers
        function net=feedforward_neural_network(net)
            L=net.num_layers;
            for k=2:L+1
                if k==2
                    net.layers{k}.State_in=net.input_data;
                    net.layers{k}.State_out=max(0,net.layers{k}.Weight*net.layers{k}.State_in+net.layers{k}.Bias);
                elseif k<=L
                    net.layers{k}.State_in=net.layers{k-1}.State_out;
                    net.layers{k}.State_out=max(0,net.layers{k}.Weight*net.layers{k}.State_in+net.layers{k}.Bias);
                elseif k==L+1
                    net.layers{k}.State_in=net.layers{k-1}.State_out;
                    net.layers{k}.State_out=net.layers{k}.Weight*net.layers{k}.State_in+net.layers{k}.Bias;
                end
            end
        end
        
        % feedforward that only gives the output without
        % instantiation
        function out=output_only_feedforward_neural_network(net,input)
            L=net.num_layers;
            out=input;
            for k=2:L+1
                if k==2
                    out=max(0,net.layers{k}.Weight*out+net.layers{k}.Bias);
                elseif k<=L
                    out=max(0,net.layers{k}.Weight*out+net.layers{k}.Bias);
                elseif k==L+1
                    out=net.layers{k}.Weight*out+net.layers{k}.Bias;
                end
            end
        end
        
        %% Block coordinate descent (BCD) updates
        % Weight update output layer
        function [W_update,b_update]=bcd_unconstrained_output_weight_regression(net)
            L=net.num_layers;
            [n_out,n_in]=size(net.layers{L+1}.Weight);
            state=net.layers{L}.State_out;
            training_output=net.output_data;
            cvx_begin quiet
            cvx_precision low
            variable W(n_out,n_in)
            variable b(n_out,1)
            minimize(norm(W*state+b*ones(1,net.num_train)-training_output,'fro'))
            cvx_end
            W_update=W;
            b_update=b;
        end
        
        % State update output layer
        function X_update=bcd_fenchel_output_state_regression(net)
            L=net.num_layers;
            n_out=net.layers{L}.OutputSize;
            m=net.num_train;
            Z=net.layers{L}.Weight*net.layers{L}.State_in+net.layers{L}.Bias;
            epsilon=net.options.FenchelBackPenalization;
            weight=net.layers{L+1}.Weight;
            Bias=net.layers{L+1}.Bias;
            training_output=net.output_data;
            
            cvx_begin quiet
            cvx_precision low
            variable X(n_out,m)
            minimize(norm(weight*X+Bias*ones(1,m)-training_output,'fro')+epsilon*norm(X-Z,'fro'))
            X>=0
            cvx_end
            X_update=X;
        end
        
        % Weights update hidden layers        
        function [W_update,b_update]=bcd_fenchel_unconstrained_hidden_weight(net,index_layer)
            State_in=net.layers{index_layer}.State_in;
            State_out=net.layers{index_layer}.State_out;
            [n_out,n_in]=size(net.layers{index_layer}.Weight);
            cvx_begin quiet
            variable W(n_out,n_in)
            variable b(n_out)
            minimize(0.5*sum(sum_square_pos(W*State_in+b*ones(1,net.num_train)))-trace(State_out'*(W*State_in+b*ones(1,net.num_train))))
            cvx_end
            W_update=W;
            b_update=b;
        end
        
        % Hidden state (out) update 
        function X_update=bcd_fenchel_hidden_state_out(net,index_layer)
            m=net.num_train;
            n_out=net.layers{index_layer}.OutputSize;
            State_in=net.layers{index_layer}.State_in;
            W=net.layers{index_layer}.Weight;
            b=net.layers{index_layer}.Bias;
            W_next=net.layers{index_layer+1}.Weight;
            b_next=net.layers{index_layer+1}.Bias;
            State_out_next=net.layers{index_layer+1}.State_out;
            epsilon=net.options.FenchelBackPenalization;
            
            cvx_begin quiet
            variable X(n_out,m)
            minimize(0.5*sum(sum_square_pos(W_next*X+b_next*ones(1,m)))-trace(State_out_next'*(W_next*X+b_next*ones(1,m)))+0.5*epsilon*sum(sum_square(X))-epsilon*trace(X'*(W*State_in+b*ones(1,m))))
            X>=0
            cvx_end
            X_update=X;
        end
        
        % Full BCD backward one step gradient update (can be seen as linearized proximal bcd)
        function net=fenchel_full_bcd_backward_one_step_gradient(net)
            L=net.num_layers;
            for k=flip(2:L+1)
                if k==L+1
                    [grad_W,grad_b,rate]=net.gradient_output_weight_regression;
                    net.layers{k}.Weight=net.layers{k}.Weight-rate*grad_W;
                    net.layers{k}.Bias=net.layers{k}.Bias-rate*grad_b;
                    net.layers{k}.State_out=net.layers{k}.Weight*net.layers{k}.State_in+net.layers{k}.Bias;
                elseif k==L
                    [grad_X,rate_X]=net.gradient_fenchel_state_output_regression;
                    net.layers{k}.State_out=max(0,net.layers{k}.State_out-rate_X*grad_X);
                    net.layers{k+1}.State_in=net.layers{k}.State_out;
                    [grad_W,grad_b,rate_W]=net.gradient_fenchel_hidden_weight(k);
                    net.layers{k}.Weight=net.layers{k}.Weight-rate_W*grad_W;
                    net.layers{k}.Bias=net.layers{k}.Bias-rate_W*grad_b;
                else
                    [grad_X,rate_X]=net.gradient_fenchel_hidden_state_out(k);
                    net.layers{k}.State_out=max(0,net.layers{k}.State_out-rate_X*grad_X);
                    net.layers{k+1}.State_in=net.layers{k}.State_out;
                    [grad_W,grad_b,rate_W]=net.gradient_fenchel_hidden_weight(k);
                    net.layers{k}.Weight=net.layers{k}.Weight-rate_W*grad_W;
                    net.layers{k}.Bias=net.layers{k}.Bias-rate_W*grad_b;
                end
            end
        end
        
        %% [Non BCD] gradient for weight, regression layer
        % full gradient computation
        function net=fenchel_full_gradient(net)
            L=net.num_layers;
            for k=flip(2:L+1)
                if k==L+1
                    [grad_W,grad_b]=net.gradient_output_weight_regression;
                    net.layers{k}.Gradients.Weight=grad_W;
                    net.layers{k}.Gradients.Bias=grad_b;
                elseif k==L
                    net.layers{k}.Gradients.State_out=net.gradient_fenchel_state_output_regression;
                    [grad_W,grad_b]=net.gradient_fenchel_hidden_weight(k);
                    net.layers{k}.Gradients.Weight=grad_W;
                    net.layers{k}.Gradients.Bias=grad_b;
                else
                    net.layers{k}.Gradients.State_out=net.gradient_fenchel_hidden_state_out(k);
                    [grad_W,grad_b]=net.gradient_fenchel_hidden_weight(k);
                    net.layers{k}.Gradients.Weight=grad_W;
                    net.layers{k}.Gradients.Bias=grad_b;
                end
            end
        end
        
        % full gradient update
        function net=fenchel_full_gradient_update(net,rate)
            net=net.fenchel_full_gradient;
            L=net.num_layers;
            for k=flip(2:L+1)
                if k==L+1
                    grad_W=net.layers{k}.Gradients.Weight;
                    grad_b=net.layers{k}.Gradients.Bias;
                    net.layers{k}.Weight=net.layers{k}.Weight_prev-rate*grad_W;
                    net.layers{k}.Bias=net.layers{k}.Bias_prev-rate*grad_b;
                    net.layers{k}.State_out=net.layers{k}.Weight*net.layers{k}.State_in+net.layers{k}.Bias;
                elseif k<=L
                    grad_X=net.layers{k}.Gradients.State_out;
                    net.layers{k}.State_out=max(0,net.layers{k}.State_out_prev-rate*grad_X);
                    net.layers{k+1}.State_in=net.layers{k}.State_out;
                    grad_W=net.layers{k}.Gradients.Weight;
                    grad_b=net.layers{k}.Gradients.Bias;
                    net.layers{k}.Weight=net.layers{k}.Weight_prev-rate*grad_W;
                    net.layers{k}.Bias=net.layers{k}.Bias_prev-rate*grad_b;
                end
            end
        end
        
        % full accelerated gradient update
        function net=fenchel_full_accelerated_gradient_update(net,rate,iter)
            L=net.num_layers;
            for k=2:L+1
                net.layers{k}.State_out=net.layers{k}.State_out+(iter-1)/(iter+2)*(net.layers{k}.State_out-net.layers{k}.State_out_prev);
                net.layers{k+1}.State_in=net.layers{k}.State_out;
                net.layers{k}.Weight=net.layers{k}.Weight+(iter-1)/(iter+2)*(net.layers{k}.Weight-net.layers{k}.Weight_prev);
                net.layers{k}.Bias=net.layers{k}.Bias+(iter-1)/(iter+2)*(net.layers{k}.Bias-net.layers{k}.Bias_prev);
            end
            net=net.fenchel_full_gradient;
            net=net.fenchel_full_gradient_update(rate);
        end
        
        % gradient weights last layer
        function [grad_W,grad_b,rate]=gradient_output_weight_regression(net)
            L=net.num_layers;
            m=net.num_train;
            if strcmp(net.options.method,'gradient') || strcmp(net.options.method,'accelerated_gradient')
                W=net.layers{L+1}.Weight_prev;
                b=net.layers{L+1}.Bias_prev;
                X=net.layers{L+1}.State_in_prev;
            else
                W=net.layers{L+1}.Weight;
                b=net.layers{L+1}.Bias;
                X=net.layers{L+1}.State_in;       
            end
            Y=net.output_data;
            grad_W=(W*X-Y+b*ones(1,m))*X';
            grad_b=m*b+(W*X-Y)*ones(m,1);
            if nargout==3
                smooth=max(m,norm(X)^2)+norm(X*ones(m,1));
                rate=1/smooth;
            end
        end
        
        % gradient last layer
        function [grad_X,rate_X]=gradient_fenchel_state_output_regression(net)
            L=net.num_layers;
            epsi=net.options.FenchelBackPenalization;
            Y=net.output_data;
            if strcmp(net.options.method,'gradient') || strcmp(net.options.method,'accelerated_gradient')
                W_nxt=net.layers{L+1}.Weight_prev;
                b_nxt=net.layers{L+1}.Bias_prev;
                W=net.layers{L}.Weight_prev;
                b=net.layers{L}.Bias_prev;
                X_out=net.layers{L}.State_out_prev;
                X_in=net.layers{L}.State_in_prev;
            else
                W_nxt=net.layers{L+1}.Weight;
                b_nxt=net.layers{L+1}.Bias;
                W=net.layers{L}.Weight;
                b=net.layers{L}.Bias;
                X_out=net.layers{L}.State_out;
                X_in=net.layers{L}.State_in;
            end
            grad_X=W_nxt'*(W_nxt*X_out+b_nxt-Y)+epsi*(X_out-W*X_in-b);
            if nargout==2
                rate_X=1/(epsi+norm(W_nxt)^2);
            end
        end
        
        % gradient weight hidden layer
        function [grad_W,grad_b,learning_rate]=gradient_fenchel_hidden_weight(net,index_layer)
            m=net.num_train;
            if strcmp(net.options.method,'gradient') || strcmp(net.options.method,'accelerated_gradient')
                W=net.layers{index_layer}.Weight_prev;
                b=net.layers{index_layer}.Bias_prev;
                X_in=net.layers{index_layer}.State_in_prev;
                X_out=net.layers{index_layer}.State_out_prev;
            else
                W=net.layers{index_layer}.Weight;
                b=net.layers{index_layer}.Bias;
                X_in=net.layers{index_layer}.State_in;
                X_out=net.layers{index_layer}.State_out;
            end
            grad_W=max(0,W*X_in+b*ones(1,m))*X_in'-X_out*X_in';
            grad_b=max(0,W*X_in+b*ones(1,m))*ones(m,1)-X_out*ones(m,1);
            if nargout==3
                smooth=max(m,norm(X_in)^2)+norm(X_in*ones(m,1));
                learning_rate=1/smooth;
            end
        end
        
        % gradient hidden state
        function [grad_X,rate_X]=gradient_fenchel_hidden_state_out(net,index_layer)
            epsi=net.options.FenchelBackPenalization;
            if strcmp(net.options.method,'gradient') || strcmp(net.options.method,'accelerated_gradient')
                W_layer_after=net.layers{index_layer+1}.Weight_prev;
                b_layer_after=net.layers{index_layer+1}.Bias_prev;
                W_layer=net.layers{index_layer}.Weight_prev;
                b_layer=net.layers{index_layer}.Bias_prev;
                X_after=net.layers{index_layer+1}.State_out_prev;
                X_out=net.layers{index_layer}.State_out_prev;
                X_in=net.layers{index_layer}.State_in_prev;
            else
                W_layer_after=net.layers{index_layer+1}.Weight;
                b_layer_after=net.layers{index_layer+1}.Bias;
                W_layer=net.layers{index_layer}.Weight;
                b_layer=net.layers{index_layer}.Bias;
                X_after=net.layers{index_layer+1}.State_out;
                X_out=net.layers{index_layer}.State_out;
                X_in=net.layers{index_layer}.State_in;
            end
            grad_X=W_layer_after'*max(0,W_layer_after*X_out+b_layer_after)-W_layer_after'*X_after+epsi*(X_out-W_layer*X_in-b_layer);
            if nargout==2
                rate_X=1/(epsi+norm(W_layer)^2);
            end
        end
        
        % corresponding f_vals
        function fval=fval_fenchel_hidden_weight(net,index_layer)
            W=net.layers{index_layer}.Weight;
            b=net.layers{index_layer}.Bias;
            X_in=net.layers{index_layer}.State_in;
            X_out=net.layers{index_layer}.State_out;
            fval=net.compute_fenchel_divergence(X_out,W*X_in+b);
        end

        function f_val=fval_fenchel_hidden_state_out(net,index_layer)
            epsi=net.options.FenchelBackPenalization;
            W=net.layers{index_layer}.Weight;
            b=net.layers{index_layer}.Bias;
            X_in=net.layers{index_layer}.State_in;
            X_out=net.layers{index_layer}.State_out;
            X_after=net.layers{index_layer+1}.State_out;
            W_after=net.layers{index_layer+1}.Weight;
            b_after=net.layers{index_layer+1}.Bias;
            f_val=net.compute_fenchel_divergence(X_after,W_after*X_out+b_after)+epsi*net.compute_fenchel_divergence(X_out,W*X_in+b);
        end
        
        %% Miscellanous
        
        % make a copy of X and W in prev
        function net=make_copies_prev(net)
            L=net.num_layers;
            for k=2:L+1
                net.layers{k}.Weight_prev=net.layers{k}.Weight;
                net.layers{k}.Bias_prev=net.layers{k}.Bias;
                net.layers{k}.State_in_prev=net.layers{k}.State_in;
                net.layers{k}.State_out_prev=net.layers{k}.State_out;
            end
        end
        
        % paste the copy of X and W 
        function net=paste_copies_prev(net)
            L=net.num_layers;
            for k=2:L+1
                net.layers{k}.Weight=net.layers{k}.Weight_prev;
                net.layers{k}.Bias=net.layers{k}.Bias_prev;
                net.layers{k}.State_in=net.layers{k}.State_in_prev;
                net.layers{k}.State_out=net.layers{k}.State_out_prev;
            end
        end
        
        % Compute Loss feedforward
        function out=compute_rmse_feedforward(net,input,true_output)
            predicted_output=net.output_only_feedforward_neural_network(input);
            out=net.compute_rmse(predicted_output,true_output);
        end
        
        % Compute fenchel objective function
        function f_val=f_val_fenchel(net)
            f_val=net.rmse_part_fenchel;
            epsi=net.options.FenchelBackPenalization;
            L=net.num_layers;
            for k=2:L
                state_out=net.layers{k}.State_out;
                state_in=net.layers{k}.State_in;
                W=net.layers{k}.Weight;
                b=net.layers{k}.Bias;
                f_val=f_val+epsi^(L+1-k)*net.compute_fenchel_divergence(state_out,W*state_in+b);
            end
        end
        
        % Compute fenchel objective function previous
        function f_val=f_val_fenchel_prev(net)
            f_val=net.rmse_part_fenchel_prev;
            epsi=net.options.FenchelBackPenalization;
            L=net.num_layers;
            for k=2:L
                state_out=net.layers{k}.State_out_prev;
                state_in=net.layers{k}.State_in_prev;
                W=net.layers{k}.Weight_prev;
                b=net.layers{k}.Bias_prev;
                f_val=f_val+epsi^(L+1-k)*net.compute_fenchel_divergence(state_out,W*state_in+b);
            end
        end
        
        % Compute total squared euclidean norm of update difference
        function out=total_norm_update_difference(net)
            L=net.num_layers;
            out=norm(net.layers{L+1}.Weight_prev-net.layers{L+1}.Weight,'fro')^2+norm(net.layers{L+1}.Bias_prev-net.layers{L+1}.Bias)^2;
            for k=2:L
                out=out+norm(net.layers{k}.State_out_prev-net.layers{k}.State_out)^2+norm(net.layers{k}.Weight_prev-net.layers{k}.Weight,'fro')^2+norm(net.layers{k}.Bias_prev-net.layers{k}.Bias)^2;
            end
        end
        
        % Compute RMSE loss from fenchel
        function out=rmse_part_fenchel(net)
            out=0.5*norm(net.output_data-net.layers{net.num_layers+1}.State_out,'fro')^2;
        end
        
        %  Compute RMSE loss from fenchel previous
        function out=rmse_part_fenchel_prev(net)
            out=0.5*norm(net.output_data-net.layers{net.num_layers+1}.State_out_prev,'fro')^2;
        end
    end
    
    methods(Static)  
        % Compute RMSE
        function out=compute_rmse(Y_1,Y_2)
            [~,m]=size(Y_1);
            out=(1/sqrt(m))*norm(Y_1-Y_2,'fro');
        end
        
        % Compute Fenchel Divergence
        function out=compute_fenchel_divergence(U,V)
            out=0.5*norm(U,'fro')^2+0.5*norm(max(0,V),'fro')^2-trace(U'*V);
        end
    end
end