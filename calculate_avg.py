import pickle
import numpy as np

def calculate_avg(syntax_list, syntax_attention):
    avg_attns = {}
    for syntaxType in syntax_list:
        temp = []
        for instance in syntax_attention:
            a = np.array(instance[syntaxType])
            if a.shape[0] != 0:
                a = a.squeeze(0)
                a = a.mean(axis=2)
                temp.append(a)
        if len(temp) != 0:
            temp = np.array(temp)
            temp = temp.reshape(temp.shape[1], temp.shape[2], temp.shape[0])
            avg_temp = temp.mean(axis=2)
            avg_attns[syntaxType] = avg_temp
    return avg_attns

# read pickle 

ast_syntax_list = ['else', 
                'if_statement', 
                'method_declaration', 
                'class_declaration', 
                'constructor_declaration',
                'return_statement']

pickle_file = pickle.load(open("/Users/jirigesi/Desktop/results/attention_results/CD4DD_syntax_attention_weights.pkl", "rb"))

# calculate average attention weights for each syntax type
avg_attns = calculate_avg(ast_syntax_list, pickle_file)

with open('CD4DD_syntax_attention_weights.pkl', 'wb') as f:
    pickle.dump(syntax_attention_weights, f)