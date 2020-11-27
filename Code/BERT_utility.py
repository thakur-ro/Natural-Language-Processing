import numpy as np
import re

from transformers import BertTokenizer, BertConfig
from transformers import BertModel
import torch
import pickle
from keras.preprocessing.sequence import pad_sequences
from BERT_config import bert_config

def get_bert_token_positions(input_text,token_list,start_from_pos=0,prior_partial_word=""):
    partial_word = ""

    pos_list = []                    

    if(prior_partial_word!=""):
        input_text = prior_partial_word + input_text 

    name_to_match = input_text.lower().replace(" ","")
    remaining_name = input_text.lower().replace(" ","")

    name = ""
    count = start_from_pos

    for entry in token_list[start_from_pos:]:
        entry_text = entry.strip("##").lower()
        if(entry_text.startswith(remaining_name) and (entry_text != remaining_name)):
            partial_word = remaining_name
            pos_list.append(count)
            break

        if(remaining_name.startswith(entry_text)):
            pos_list.append(count)
            remaining_name = remaining_name[len(entry_text):]
            name = name + entry_text
            if(name == name_to_match):
                break
        else:
            pos_list = []
            name = ""
            remaining_name = name_to_match
            if(remaining_name.startswith(entry_text)):                                 
                pos_list.append(count)                                                                
                remaining_name = remaining_name[len(entry_text):]
                name = name + entry_text   
                if(name == name_to_match):                                   
                    break

        count = count + 1

    return [pos_list,partial_word]

class BERT_utility:
    def __init__(self, get_encodings = False, get_embeddings = True, use_finetuned_model=False):
        
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_config['ncbi_base_path'])
        
        if(use_finetuned_model):
            self.model_in_use = "FINETUNED"
            self.bert_model = pickle.load(open(bert_config['entity_finetuned_model_path'],"rb"))
            self.bert_model.cpu()
        else:
            self.model_in_use = "BASE"
            self.bert_model = BertModel.from_pretrained(bert_config['ncbi_base_path'])
            
        self.get_encodings = get_encodings
        self.get_embeddings = get_embeddings
    
    def get_embeddings_for_nsp(self,first_sentence, second_sentence):
        encoding = self.bert_tokenizer(first_sentence, second_sentence, return_tensors='pt')['input_ids'][0]

        return encoding

    def process_string_finetune(self, string_input, padding_length):
        sentences = string_input.split("\n")

        positions_covered = 0

        word_list = list()
        
        if(self.get_encodings):
            encoding_list = list()

        for index in range(len(sentences)):
            sentence = sentences[index]
            start_index_bert = max(0,index-padding_length)
            end_index_bert = min(len(sentences),index+padding_length)

            bert_input = ' '.join(sentences[start_index_bert:(end_index_bert+1)])

            encodings = self.bert_tokenizer.encode(bert_input,add_special_tokens = True)
            
            if(len(encodings)>=512):
                encodings = encodings[0:512]
            
            if(self.get_encodings):
                encoding_list.append(encodings)
            
            input_ids = torch.tensor(encodings).long().unsqueeze(0)
            
            if(self.get_embeddings):
                bert_vector = self.bert_model(input_ids,token_type_ids=None)

            bert_tokens = self.bert_tokenizer.convert_ids_to_tokens(encodings) 

            start_pos = 0
            prior_pos = get_bert_token_positions(' '.join(sentences[start_index_bert:index]),bert_tokens)[0]

            if(len(prior_pos)>0):
                start_pos = max(prior_pos)

            words = sentence.split()

            sentence_covered = ''

            prior_partial_word = ''

            word_index = 0

            for current_word in words:
                new_dict = {}

                new_dict["word"] = current_word

                [bert_token_positions, partial_word] = get_bert_token_positions(current_word,bert_tokens,start_pos,prior_partial_word)

                vec_list_layers = []

                if(len(bert_token_positions)==0):
                    prior_partial_word = ""
                    word_index = word_index + 1
                    continue

                if(partial_word != ""):
                    prior_partial_word = partial_word
                    start_pos = bert_token_positions[-1]
                else:
                    prior_partial_word = ""
                    start_pos = bert_token_positions[-1] + 1

                token_position = string_input.find(current_word, positions_covered)

                spaces_between = string_input[positions_covered:token_position]

                sentence_covered = sentence_covered + spaces_between + current_word

                positions_covered = token_position + len(current_word)
                
                if(self.get_embeddings):
                    vec_list = []
                    
                    for entry in bert_token_positions:
                        if(self.model_in_use == "FINETUNED"):
                            vec_list.append(bert_vector[1][12][0][entry].data.numpy())
                        else:
                            vec_list.append(bert_vector[0][0][entry].data.numpy())

                    vec_word = np.mean(vec_list,axis=0)

                    new_dict["keyword_vector"] = vec_word
                
                new_dict["sentence_index"] = index + 1

                new_dict["word_index"] = word_index
                
                if(self.get_encodings):
                    new_dict["bert_token_positions"] = bert_token_positions

                word_list.append(new_dict)

                word_index = word_index + 1
        
        if(self.get_encodings):
            self.encoding_list = encoding_list

        return word_list 