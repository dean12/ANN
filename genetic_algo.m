
function nn_structures = genetic_algo(initial_population,train_data,test_data)
%Genetic Algorithm 

%Script that runs the genetic algorithm to choose the neural network
%structure that best fits the data. 

%% 1. Set up the initial population 
% Neural networks are restricted to 10 layers and may not have more than
% 100 nodes in ech layer. 

%Generate a random number of nodes for each layer for each initial neural 
%network.

%maximum number of hidden layers
max_length = 4;
ga_iterations = 30;

all_networks = cell(initial_population,2);  
    
    for o=1:length(all_networks)
        all_networks{o,1} = [25 , randi([1 100],1,max_length), 10];
        
    end 
 
    
%Record best performers
best_performers = zeros(1,ga_iterations) ;
 

% Loop starts here

for q=1:ga_iterations
    
%% 2. Train Neural Networks and evaluate fitness



disp(q) 
 

net_struc = all_networks(:,1); 

poolobj =  parpool('AttachedFiles',{'feedforward.m','hidden_deltas.m','f_gradient.m','neuralnet_genetic.m'});

    parfor y = 1:length(all_networks)
        
        all_networks{y,2} = neuralnet_genetic(train_data,test_data,net_struc{y})
         
        
    end  
 
    
nn_structures = all_networks; 

 delete(poolobj)
best_performers(q) = max([all_networks{:,2}]); 

    
%% 3. Rank population, discard lowest '1/2' and create children to replace them

all_networks = sortrows(all_networks,2); 

percentage_kill = 0.5;

%Discard bottom quarter 
all_networks = all_networks((round(percentage_kill*(length(all_networks))):length(all_networks)),:);

% Calculate number of missing networks 
length_diff = initial_population -length(all_networks);

%Create new cell to house new networks
new_nn = cell(length_diff,2);

%Create required children
    for o=1:length_diff       
        crossover_point = randi([2 max_length]);
        
        %Sample 2 without replacement. 
        
        parents = randsample(all_networks(:,1),2) ;
        parent_a = parents{1} ;
        parent_b = parents{2} ;        
        
        %Take on portions from parents in 'all_network'
        new_nn = {[parent_a(1:crossover_point), parent_b(crossover_point+1:length(parent_b))], 0}; 
        all_networks = [all_networks; new_nn] ;
    end 
    
    

    

 
 
 
 
end


 plot(best_performers) 
 


end 



   



