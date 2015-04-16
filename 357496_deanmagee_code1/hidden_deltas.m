function output = hidden_deltas(weights,biases,structure, delta_f,node_datainputs,section_size)
%Computes the deltas for each layer. 

% Place final layer's delta in last element of cell
all_deltas = cell(1,length(structure)) ;
all_deltas{length(all_deltas)} = delta_f;


% Calculate intermediate values of deltas
for l=(length(structure)-1):(-1):2
    all_deltas{l} = (weights{l+1}' * all_deltas{l+1}')' .* f_gradient(node_datainputs{l} * weights{l}' + repmat((biases{l})',section_size,1)); 
    
    
end


output = all_deltas;

end
