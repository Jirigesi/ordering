import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import  RobertaConfig, RobertaModel, RobertaTokenizer
import argparse
import json
import os
from model2 import Model
import random
import multiprocessing
from tqdm import tqdm, trange
import numpy as np
import javalang
from tree_sitter import Language, Parser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
np.random.seed(0)
import seaborn as sns
import collections
import pickle
import sklearn
from matplotlib import cm
from sklearn import manifold
from tree_sitter import Language, Parser
import javalang
import numpy as np 

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label


def convert_examples_to_features(js,tokenizer):
    #source
    code=' '.join(js['func'].split())
    code_tokens=tokenizer.tokenize(code)[:block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    
    return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)  

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def traverse(code, node,depth=0):
    declaration = {}
    stack = []
    stack.append(node)
    while stack:
        node = stack.pop()
        if ('declaration' in node.type and node.type != "local_variable_declaration") or 'if_statement' in node.type or 'else' in node.type or 'return' in node.type:
            data = code[node.start_byte:node.end_byte].split('{')[0].strip().split(' ')
            if node.type in declaration:
                declaration[node.type].append(data)
            else:
                declaration[node.type] = [data]
        for child in node.children:
            stack.append(child)
    return declaration

def label_tokens(token_list, declaration):
    types = [] 
    for token in token_list:
        flag = False
        for key in declaration:
            for value in declaration[key]:
                if token in value:
                    types.append(key)
                    flag = True
                    break
            if flag:
                break
        if not flag:
            types.append("other")
    return types

def get_extended_types(token_list, types):
    tree = list(javalang.tokenizer.tokenize(" ".join(token_list)))
    code = ' '.join(token_list)
    right = 0
    left = 0
    postion_mapping = [] 

    while right < len(code):
        if code[right] == ' ':
            postion_mapping.append((left, right))
            left = right + 1
        right += 1

    # add the last token
    postion_mapping.append((left, right))
    code = ["<s>"]
    extended_types = []
    left = 0
    for node in tree:
        # rewrite code
        node = str(node).split(' ')
        if node[1] == '"MASK"':
            code.append('<mask>')
        else:
            code.append(node[1][1:-1])
        # extend types
        left = int(node[-1]) -1
        right = left + len(node[1][1:-1])
        # check (left, right) in postion_mapping and get the index
        for i in range(len(postion_mapping)):
            if left >= postion_mapping[i][0] and right <= postion_mapping[i][1]:
                extended_types.append([types[i], node[1]])
                break
    code.append("</s>")
    return extended_types, ' '.join(code)


def get_ast_types(code):
    code = code.replace("{", " {")
    code = " ".join(code.split())
    code_list = code.split(' ')
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    
    declaration = traverse(code, root_node)
    types = label_tokens(code_list, declaration)

    ast_types, rewrote_code = get_extended_types(code_list, types)
    # check the index of first second value is the "{"
    if ast_types[0][1] == '"class"':
        return ['[CLS]'] + [i[0] for i in ast_types] + ['[SEP]'], rewrote_code
    index_ = 0
    # if not class declaration, find the first "{" and add method_declaration before it
    for i in range(len(ast_types)):
        if ast_types[i][1] == '"{"':
            index_ = i
            break
    final_types = [] 
    final_types.append('[CLS]')
    for i in range(len(ast_types)):
        if i < index_:
            final_types.append("method_declaration")
        else:
            final_types.append(ast_types[i][0])
    final_types.append('[SEP]')
    return final_types, rewrote_code

def get_start_end_of_token_when_tokenized(code_list, types, tokenizer):
  reindexed_types = []
  start = 0
  end = 0
  for each_token in code_list: 
      tokenized_list = tokenizer.tokenize(each_token)
      end += len(tokenized_list)
      reindexed_types.append((start, end-1))
      start = end
  return reindexed_types

def getSyntaxAttentionScore(model, data, tokenizer, syntaxList):
    block_size = 400
    all_instances = []
    number = 0
    fails = 0 
    for code_sample in tqdm(data):
        number += 1
        Instantce_Result = {}
        try: 
            for syntaxType in syntaxList:
                Instantce_Result[syntaxType] = []


            types_1, rewrote_code_1 = get_ast_types(code_sample[1])
            
            tokenized_ids_1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(rewrote_code_1))

            if len(tokenized_ids_1) > 400:
                tokenized_ids_1 = tokenized_ids_1[:399] + [tokenizer.sep_token_id]
            
            padding_length = block_size - len(tokenized_ids_1)
            tokenized_ids_1+=[tokenizer.pad_token_id]*padding_length
            labels = code_sample[0]

            source_ids = torch.tensor(tokenized_ids_1).unsqueeze(0).to(device)
            labels = torch.tensor(labels).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(source_ids,labels)

            _attention = output[2].attentions
            start_end = get_start_end_of_token_when_tokenized(rewrote_code_1, types_1, tokenizer)
            
            for syntaxType in syntaxList:
                attention_weights = [[[] for col in range(12)] for row in range(12)]
                for layer in range(12):
                    for head in range(12):
                        for each_sep_index in np.where(types_1==syntaxType)[0]:
                            start_index, end_index = start_end[each_sep_index]
                            interim_value = _attention[layer][0][head][:, start_index:end_index+1].mean().cpu().detach().numpy()
                            if np.isnan(interim_value):
                                pass
                            else: 
                                attention_weights[layer][head].append(interim_value)     
                if np.array(attention_weights).shape[2] != 0:
                    Instantce_Result[syntaxType].append(np.array(attention_weights))
            all_instances.append(Instantce_Result)
        # if get error, print the error and continue
        except Exception as e:
            print('Error index:', number)
            fails += 1
            continue
            
    print("fails: ", fails)
    return all_instances

Language.build_library(
	# Store the library in the `build` directory
	'build/my-languages.so',	
	# Include one or more languages
	[
		'/Users/jirigesi/Documents/tree-sitter-java'
	]
)
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
parser = Parser()

parser.set_language(JAVA_LANGUAGE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = RobertaConfig.from_pretrained('microsoft/codebert-base')

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaModel.from_pretrained('microsoft/codebert-base',
                                    output_attentions=True, 
                                    output_hidden_states=True)

model=Model(model,config,tokenizer)

checkpoint_prefix = "saved_models/CD4DD/model_CD.bin"

model.load_state_dict(torch.load(checkpoint_prefix, map_location=torch.device('cpu')))

model = model.to(device)

block_size = 400
test_data_file = "../dataset/test.jsonl"

data = []
with open(test_data_file) as f:
    for line in f:
        js=json.loads(line.strip())
        data.append((js['target'], ' '.join(js['func'].split())))

syntax_list = ['else', 
                'if_statement', 
                'method_declaration', 
                'class_declaration', 
                'constructor_declaration',
                'return_statement']

syntax_attention_weights = getSyntaxAttentionScore(model, data, tokenizer, syntax_list)

# pickle the result
import pickle
with open('CD4DD_syntax_attention_weights.pkl', 'wb') as f:
    pickle.dump(syntax_attention_weights, f)