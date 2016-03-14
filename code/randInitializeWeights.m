function W = randInitializeWeights(L_in, L_out)
% W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
% of a layer with L_in incoming connections and L_out outgoing 
% connections. 

% Randomly initialize the weights to small values
epsilon_init = sqrt(6)/sqrt(L_in+L_out+1);
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
