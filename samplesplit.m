%%%% Split data into subtrain and subtest sets



train = csvread('train.csv'); 

subtrain= train(1:50000,:);

subtest= train(50001:60000,:);

test = csvread('test-nolabel.csv')