% This takes in the inputs and feeds it through the network.
% Outputs a vector that contains the outputs of each sample fed through the network 

function [final_output, node_datainputs]  = feedforward(weights,biases,a,section_size,structure)

% contains all observations in this batch


pre_activ = a(:,2:26);
node_datainputs=cell(1,length(structure)); 
node_datainputs{2}=pre_activ;

for l = 2:(length(weights)-1)
    pre_activ = pre_activ * weights{l}' + repmat((biases{l})',section_size,1);
    % post_activ = tanh(pre_activ) ;
    node_datainputs{l+1} = tanh(pre_activ); 
    
end
final_output = tanh(pre_activ);
end