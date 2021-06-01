import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
import torch
from find_index import find_diff, find_phrase_start, find_phrase_end, find_prepos_start, find_original_mask, mask_start_end, mask_start_end_pdep

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

loc = read_data('data/Ferretti01.txt')
fer = read_data('data/Ferretti07.txt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_base = BertForMaskedLM.from_pretrained('bert-base-uncased')
ids2token = tokenizer.convert_ids_to_tokens
token2ids = tokenizer.convert_tokens_to_ids

#tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
#bert_large = BertForMaskedLM.from_pretrained('bert-large-uncased')

def return_masked_tokens(arr1):
    arr1_token = ids2token(arr1)
    ########
    mask_index = find_original_mask(arr1_token)
    phrase_start = find_phrase_start(arr1_token)
    phrase_end = find_phrase_end(arr1_token)
    prepos_start = find_prepos_start(arr1_token)
    ########
    standard = arr1_token
    standard_prepdep = mask_start_end_pdep(arr1_token,mask_index-2,mask_index-1)
    context_temp = mask_start_end(arr1_token,1,phrase_start-1)
    context_phrase_prepos = mask_start_end(context_temp,prepos_start+1,mask_index-1)
    context_phrase = mask_start_end(context_temp,prepos_start,mask_index-1)
    if phrase_end+1 == prepos_start:
        context_prepos = context_phrase_prepos
        context = context_phrase
    else:
        context_prepos = mask_start_end(context_phrase_prepos,phrase_end+1,prepos_start-1)
        context = mask_start_end(context_phrase,phrase_end+1,prepos_start-1)
    return standard, standard_prepdep, context_phrase_prepos, context_phrase, context_prepos, context, mask_index

def calc_surp(seq,val):
    seq = seq.detach().numpy()
    val = val.detach().numpy()
    exp_sum = sum(np.exp(seq))
    portion = np.exp(val) / exp_sum
    surp = - np.log(portion)
    return surp

def pair_prob_all(s1,arr,s2,mask,model):
    inputs  = tokenizer(s1, return_tensors="pt")
    indice  = token2ids(arr)
    inputs["input_ids"] = torch.tensor([indice])
    labels  = tokenizer(s2, return_tensors="pt")["input_ids"]
    label   = labels[0][int(mask)]
    if len(inputs["input_ids"][0]) == len(labels[0]):
        outputs = model(**inputs, labels=labels)
        loss    = outputs[0]
        logits  = outputs[1]
        probs   = logits[0][int(mask)]
        prob    = logits[0][int(mask)][int(label)]
        surp    = calc_surp(probs,prob)
        return loss, logits, surp
    else:
        return 0, 0, 0

def return_surprisal_all(corpus,model):
    corpus_all = []
    for i in range(len(corpus)):
        indexed = tokenizer(corpus[i][1], return_tensors="pt")["input_ids"][0]
        standard, stand_pdep, cphpr, cph, cpr, con, mask = return_masked_tokens(indexed)
        candidates = [standard, stand_pdep, cphpr, cph, cpr, con]
        candidate_names = ['standard', 'standard_prepdep','con_phrase_prep', 'con_phrase', 'con_prep', 'context']
        fillers = corpus[i][3].split(' ')
        for cand in range(len(candidates)):
            temp = []
            temp.append(corpus[i][0])
            temp.append(candidate_names[cand])
            temp.append(candidates[cand])
            print(candidates[cand])
            temp.append(corpus[i][2])
            for j in range(len(fillers)):
                text = corpus[i][1].replace('[MASK]',fillers[j])
                print(text)
                loss, logits, surp = pair_prob_all(corpus[i][1], candidates[cand], text, mask, model)
                if loss != 0:
                    scores = '('+fillers[j]+', '+str(surp)+')'
                    temp.append(scores)
                else:
                    scores = '('+fillers[j]+', '+'TOKENIZATION ERROR'+')'
                    temp.append(scores)
            corpus_all.append(temp)
    return corpus_all

loc_res_all = return_surprisal_all(loc, bert_base)
fer_res_all = return_surprisal_all(fer, bert_base)

file_aspect_loc_all_stan  = open('results/fer01_result_standard.txt','w')
file_aspect_loc_all_stanpdep  = open('results/fer01_result_standard_pdep.txt','w')
file_aspect_loc_all_cphpr = open('results/fer01_result_con_phrase_prep.txt','w')
file_aspect_loc_all_cph   = open('results/fer01_result_con_phrase.txt','w')
file_aspect_loc_all_cpr   = open('results/fer01_result_con_prep.txt','w')
file_aspect_loc_all_con   = open('results/fer01_result_context.txt','w')

file_aspect_fer_all_stan  = open('results/fer07_result_standard.txt','w')
file_aspect_fer_all_stanpdep  = open('results/fer07_result_standard_pdep.txt','w')
file_aspect_fer_all_cphpr = open('results/fer07_result_con_phrase_prep.txt','w')
file_aspect_fer_all_cph   = open('results/fer07_all_result_con_phrase.txt','w')
file_aspect_fer_all_cpr   = open('results/fer07_result_con_prep.txt','w')
file_aspect_fer_all_con   = open('results/fer07_result_context.txt','w')

files_loc = [file_aspect_loc_all_stan, file_aspect_loc_all_stanpdep, file_aspect_loc_all_cphpr, file_aspect_loc_all_cph,
             file_aspect_loc_all_cpr, file_aspect_loc_all_con]

files_fer = [file_aspect_fer_all_stan, file_aspect_fer_all_stanpdep, file_aspect_fer_all_cphpr, file_aspect_fer_all_cph,
             file_aspect_fer_all_cpr, file_aspect_fer_all_con]

def write_cases(corpus,wfiles):
    for i in range(len(corpus)):
        wfile = wfiles[int(i%6)]
        for j in range(len(corpus[i])):
            wfile.write(str(corpus[i][j])+'\t')
            if j == len(corpus[i])-1:
                wfile.write('\n')

write_cases(loc_res_all,files_loc)
write_cases(fer_res_all,files_fer)
