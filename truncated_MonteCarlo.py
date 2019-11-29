"""
Created on Fri Nov  29 18:04:13 2019
A script to  run simulations with the truncated Monte-Carlo method 

@author: Thomas-Galtier
"""
 
import numpy as np 
import scenario

from scipy.stats import norm


def truncated_MC(preprocessed_node_list, sv_accuracy=0.001, contrib_accuracy=0.01):
    """Return the vector of approximated shapeley value corresponding to a list of node and a characteristic function using the truncated monte-carlo method."""
    
    characteristic_func=fl_training.compute_test_score
    n = len(preprocessed_node_list)
    
    # we store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict={():0} # the dictionary that will countain the values
    listn=np.arange(n)
    
    # return the characteristic function of the nodelist associated to the ensemble  permut, without recomputing it if it was already computed
    def not_twice_characteristic(permut):
        #sort permut
        isin=np.repeat(False,n)
        isin[permut]=True
        permut=listn[isin]
        try: #return the characteristic_func(permut) if it was already computed
            return char_value_dict[tuple(permut)]
        except KeyError: #characteristic_func(permut) has not been computedyet, so we compute, store, and return characteristic_func(permut)
            small_node_list = [preprocessed_node_list[i] for i in permut]
            char_value_dict[tuple(permut)]= characteristic_func(small_node_list)
            return char_value_dict[tuple(permut)]

        
    # truncated_MC initialisation
    if n==1:
        return np.array([1])
    else:
        characteristic_all_node= not_twice_characteristic(np.arange(n)) #characteristic function on all nodes
        sv = np.zeros(n) #store current estimation of the shapley values 
        mean_squares_sv = np.zeros(n) # used for variance computation in the condition of the while loop below 
        permutation=np.zeros(n) # store the current permutation 
        firsts_are_equal =True
        t=0
        while t<50 or t<=norm.ppf(0.975, loc=0, scale=1)**2* np.max(mean_squares_sv-sv**2) /sv_accuracy**2 or firsts_are_equal: #check if the length of the confidence interval  is below the value of sv_accuracy
            t+=1
            print(t<50,t<=norm.ppf(0.975, loc=0, scale=1)**2* np.max(mean_squares_sv-sv**2) /sv_accuracy**2,firsts_are_equal)
            print(t)
            print(sv)
            previous_sv = np.copy(sv)#store previous estimation of the shapley values
            previous_mean_square_sv= np.copy(mean_squares_sv) #store previous value of   the mean_square_sv
            previous_permutation=np.copy(permutation)# store the previous permutation
            permutation = np.random.permutation(n) # store the current permutation
            if firsts_are_equal and t>1:# makes sure we don't stop the loop because the first permutations are the same (very unlikely but can still happen if n is small)
                firsts_are_equal=   (permutation==previous_permutation).all()
            char_nodelists = np.zeros(n+1) # store the characteristic function on each ensemble built with the first elements of the permutation
            char_nodelists[-1]=characteristic_all_node
            for j in range(n):  # iteration on sv[permutation[j]]
                if abs(characteristic_all_node-char_nodelists[j])<contrib_accuracy :
                    char_nodelists[j+1] = char_nodelists[j]
                else:
                    char_nodelists[j+1] = not_twice_characteristic(permutation[:j+1])
                sv[permutation[j]]              = (t-1)/t*previous_sv[permutation[j]]            +(char_nodelists[j+1]-char_nodelists[j])/t
                mean_squares_sv[permutation[j]] = (t-1)/t*previous_mean_square_sv[permutation[j]]+(char_nodelists[j+1]-char_nodelists[j])**2/t
    return({'sv':sv,'std': np.sqrt((mean_squares_sv-sv**2)/t), 'all_val':char_value_dict})
        
                
# TEST:        
#import data_splitting
#import fl_training
#my_custom_scenario = scenario.Scenario()
#my_custom_scenario.nodes_count=5
#my_custom_scenario.amounts_per_node = [1/my_custom_scenario.nodes_count for i in range ((my_custom_scenario.nodes_count))]
#my_custom_scenario.samples_split_option = 'Random' # or 'Stratified'
#my_custom_scenario.testset_option = 'Centralised' # Toggle between 'Centralised' and 'Distributed'
#my_custom_scenario.x_train = my_custom_scenario.x_train[:5000] # Truncate dataset if needed for quicker debugging/testing
#my_custom_scenario.y_train = my_custom_scenario.y_train[:5000] # Truncate dataset if needed for quicker debugging/testing
#my_custom_scenario.x_test = my_custom_scenario.x_test[:1000] # Truncate dataset if needed for quicker debugging/testing
#my_custom_scenario.y_test = my_custom_scenario.y_test[:1000] # Truncate dataset if needed for quicker debugging/testing
#
#node_list = data_splitting.process_data_splitting_scenario(my_custom_scenario)
#preprocessed_node_list = fl_training.preprocess_node_list(node_list)
#res1=truncated_MC(preprocessed_node_list, sv_accuracy=0.005, contrib_accuracy=0.001)
#
#char_value_list=[   res1['all_val'][tuple(i)]  for r in range(1,my_custom_scenario.nodes_count+1) for i in combinations(range(my_custom_scenario.nodes_count),r)]
#res1_true=main(my_custom_scenario.nodes_count,char_value_list)
#print(abs(res1['sv']-res1_true) <0.005)#vrai 95% du temps



        

        
