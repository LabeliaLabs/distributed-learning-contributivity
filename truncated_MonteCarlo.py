 
import numpy as np 
import scenario
import data_splitting
import fl_training
import operator as op 

from functools import reduce


#my_custom_scenario = scenario.Scenario()
#my_custom_scenario.nodes_count=5
#my_custom_scenario.amounts_per_node = [1/my_custom_scenario.nodes_count for i in range ((my_custom_scenario.nodes_count))]
#my_custom_scenario.samples_split_option = 'Random' # or 'Stratified'
#my_custom_scenario.testset_option = 'Centralised' # Toggle between 'Centralised' and 'Distributed'
#my_custom_scenario.x_train = my_custom_scenario.x_train[:3000] # Truncate dataset if needed for quicker debugging/testing
#my_custom_scenario.y_train = my_custom_scenario.y_train[:3000] # Truncate dataset if needed for quicker debugging/testing
#my_custom_scenario.x_test = my_custom_scenario.x_test[:1000] # Truncate dataset if needed for quicker debugging/testing
#my_custom_scenario.y_test = my_custom_scenario.y_test[:1000] # Truncate dataset if needed for quicker debugging/testing
#
#node_list = data_splitting.process_data_splitting_scenario(my_custom_scenario)
#preprocessed_node_list = fl_training.preprocess_node_list(node_list)
#


def truncated_MC(preprocessed_node_list, sv_accuracy=0.0001, contrib_accuracy=0.01):
    """Return the vector of approximated shapeley value corresponding to a list of node and a characteristic function using the truncated monte-carlo method."""
    
    characteristic_func=fl_training.compute_test_score
    n = len(preprocessed_node_list)
    
    # we store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict={} # the dictionary that will countain the values
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
        previous_sv = np.ones(n)
        sv = np.zeros(n)
        t=0
        characteristic_all_node= not_twice_characteristic(np.arange(n))
        previous_permutation=np.zeros(n)
        firsts_are_equal =True
        
        while (abs(sv-previous_sv)>sv_accuracy).any() or firsts_are_equal: # actualisation of sv'svalue
            t+=1
            previous_sv = np.copy(sv)
            permutation = np.random.permutation(n)
            if firsts_are_equal:# makes sure we don't stop the loop because the first permutations are the same
                if t==1:
                    previous_permutation=np.copy(permutation)
                else:
                    firsts_are_equal=   (permutation==previous_permutation).all()
                    previous_permutation=np.copy(permutation)
            char_nodelists = np.zeros(n+1)
            char_nodelists[-1]=characteristic_all_node
            for j in range(n):  # iteration on sv[permutation[j]]
                if abs(characteristic_all_node-char_nodelists[j])<contrib_accuracy :
                    char_nodelists[j+1] = char_nodelists[j]
                else:
                    char_nodelists[j+1] = not_twice_characteristic(permutation[:j+1])
                sv[permutation[j]] = (t-1)/t*previous_sv[permutation[j]]+(char_nodelists[j+1]-char_nodelists[j])/t
    print(t)
    return(sv)
        
                
        
#res1=truncated_MC(preprocessed_node_list)
#res2=truncated_MC(preprocessed_node_list)
#res3=truncated_MC(preprocessed_node_list)
#res4=truncated_MC(preprocessed_node_list)
#print(res1, "\n",res2, "\n",res3, "\n",res4, "\n",)
# [0.04678409 0.07898474 0.08309079 0.06642378 0.10971659] 
# [0.09454124 0.11483221 0.06273904 0.10528716 0.10256266] 
# [0.0808666  0.12472028 0.04013229 0.1309129  0.09346555] 
# [0.11298114 0.10456733 0.0888828  0.08594821 0.10383444]  
        
        
        
