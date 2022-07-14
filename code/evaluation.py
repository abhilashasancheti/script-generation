import os
import sacrebleu
import numpy as np
import ast
from bs4 import BeautifulSoup
import re
import random
import copy
import csv
import json
from sacremoses import MosesTokenizer, MosesDetokenizer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
chencherry = SmoothingFunction()

NUM_SEQUENCES = 5
fold_dict={}
fold_dict[1] = ['baking a cake','borrowing a book from the library', 'flying in an airplane', 'going on a train', 'riding on a bus']
fold_dict[2] = ['getting a hair cut','going grocery shopping','planting a tree','repairing a flat bicycle tire','taking a bath']
fold_dict[3] = ['eating in a fast food restaurant', 'paying with a credit card', 'playing tennis', 'going to the theater','taking a child to bed']
fold_dict[4] = ['washing dishes', 'making a bonfire', 'going to the sauna', 'making coffee', 'going to the swimming pool']
fold_dict[5] = ['taking a shower', 'ironing laundry', 'taking a driving lesson', 'going to the dentist', 'going to a funeral']
fold_dict[6] = ["washing one's hair", 'fueling a car', 'sending food back (in a restaurant)', 'changing batteries in an alarm clock', 'checking in at an airport']
fold_dict[7] = ['having a barbecue', 'ordering a pizza', 'cleaning up a flat', 'making scrambled eggs', 'taking the underground']
fold_dict[8] = ['renovating a room', 'cooking pasta', 'sewing a button', 'doing laundry', 'going bowling']


dict_script = {}
dict_script["bake a cake"]="baking a cake"
dict_script["borrow a book from the library"]="borrowing a book from the library"
dict_script["change batteries in an alarm clock"]="changing batteries in an alarm clock"
dict_script["fly in an airplane"]="flying in an airplane"
dict_script["get a hair cut"]="getting a hair cut"
dict_script["go grocery shopping"]="going grocery shopping"
dict_script["go on a train"]="going on a train"
dict_script["plant a tree"]="planting a tree"
dict_script["repair a flat bicycle tire"]="repairing a flat bicycle tire"
dict_script["ride on a bus"]="riding on a bus"
dict_script["take a bath"]="taking a bath"
dict_script["eat in a fast food restaurant"]="eating in a fast food restaurant"
dict_script["pay with a credit card"]="paying with a credit card"
dict_script["play tennis"]="playing tennis"
dict_script["go to the theater"]="going to the theater"
dict_script["take a child to bed"]="taking a child to bed"
dict_script["wash dishes"]="washing dishes"
dict_script["make a bonfire"]="making a bonfire"
dict_script["go to the sauna"]="going to the sauna"
dict_script["make coffee"]="making coffee"
dict_script["go to the swimming pool"]="going to the swimming pool"
dict_script["take a shower"]="taking a shower"
dict_script["iron laundry"]="ironing laundry"
dict_script["take a driving lesson"]="taking a driving lesson"
dict_script["go to the dentist"]="going to the dentist"
dict_script["go to a funeral"]="going to a funeral"
dict_script["wash one's hair"]="washing one's hair"
dict_script["fuel a car"]="fueling a car"
dict_script["send food back (in a restaurant)"]="sending food back (in a restaurant)"
dict_script["check in at an airport"]="checking in at an airport"
dict_script["have a barbecue"]="having a barbecue"
dict_script["order a pizza"]="ordering a pizza"
dict_script["clean up a flat"]="cleaning up a flat"
dict_script["make scrambled eggs"]="making scrambled eggs"
dict_script["take the underground"]="taking the underground"
dict_script["renovate a room"]="renovating a room"
dict_script["cook pasta"]="cooking pasta"
dict_script["sew a button"]="sewing a button"
dict_script["do laundry"]="doing laundry"
dict_script["go bowling"]="going bowling"

def read_finetuned(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        hypotheses = []
        for line in lines:
            if not line.startswith("=="):
                hypotheses.append(line.strip(' ').rstrip(' <EOS>'))
    return hypotheses

def read_pretrained(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        hypotheses = []
        i=0
        l=0
        for line in lines:
            if len(line.split())==0:
                continue
            elif not line.startswith("==") and i>1:
                hypotheses.append(line.strip())
                i-=1
            elif not line.startswith("==") and i<=1:
                hypotheses[-1] = hypotheses[-1] +" "+ line.strip()
            else:
                i+=1
            if i==3:
                i=2
    return hypotheses


def group_hypotheses(hypotheses):
    hypotheses_basic = []
    num_inputs = len(hypotheses)//NUM_SEQUENCES
    for i in range(NUM_SEQUENCES):
        hypos = []
        for j in range(num_inputs):
            hypos.append(hypotheses[i+(j*NUM_SEQUENCES)].strip().split(':')[1].lstrip())
        hypotheses_basic.append(hypos)
    return hypotheses_basic

def group_ilm_hypotheses(hypotheses):
    hypotheses_basic = []
    num_inputs = len(hypotheses)//NUM_SEQUENCES
    for i in range(NUM_SEQUENCES):
        hypos = []
        for j in range(num_inputs):
            hypos.append(hypotheses[i+(j*NUM_SEQUENCES)].strip().split('<SEP>')[1].lstrip())
        hypotheses_basic.append(hypos)
    return hypotheses_basic

def convert_tokens_to_basic(hypotheses_basic):
    # convert tokens into a list
    hypotheses_modified = []
    for hyp in hypotheses_basic:
        hypo = []
        for scenario in hyp:
            scenario = scenario.replace('<EEVENT>','</bevent>').replace('<BEVENT>','<bevent>')
            soup = BeautifulSoup(scenario)
            #print(soup, scenario)
            events = []
            for a in soup.find_all('bevent'):
                events.append(a.string)
            if len(events)>0:
                h = ""
                idx=0
                for e in events:
                    if e is not None:
                        h+= str(idx+1) + ". " + e.strip() + " "
                        idx+=1
                hypo.append(h.strip())
        hypotheses_modified.append(hypo) 
    return hypotheses_modified

def convert_tokens_to_basic_ungrouped(hypotheses_basic):
    # convert tokens into a list
    hypotheses_modified = []
    for scenario in hypotheses_basic:
        scenario = scenario.replace('<EEVENT>','</bevent>').replace('<BEVENT>','<bevent>')
        soup = BeautifulSoup(scenario)
        #print(soup, scenario)
        events = []
        for a in soup.find_all('bevent'):
            events.append(a.string)
        if len(events)>0:
            h = ""
            idx=0
            for e in events:
                if e is not None:
                    h+= str(idx+1) + ". " + e.strip() + " "
                    idx+=1
            hypotheses_modified.append(h.strip()) 
    return hypotheses_modified


def read_references(file_path='./data/valid_references.txt'):
    with open(file_path) as f:
        lines = f.readlines()
        references = []
        for line in lines:
            ref = []
            x = ast.literal_eval(line)
            for i in x:
                ref.append(" ".join(i))
            references.append(ref)
    return references

def read_ilm_references(file_path='./data/valid_inference_references.txt'):
    with open(file_path) as f:
        lines = f.readlines()
        references = []
        for line in lines:
            references.append(line.strip().strip('<EOS>').strip())
    return references

def convert_refs_for_bleu(references):
    new_references = []
    for ref in references:
        for idx, i in enumerate(ref):
            if len(new_references) < 50:
                new_references.append([i])
            else:
                new_references[idx].append(i)
    return new_references

def replace_blanks(hypotheses):
    new_hypotheses = []
    for hyp in hypotheses:
        splitted_hyp = hyp.strip().split('<SEP> ')
        scenario = splitted_hyp[0]
        answer = splitted_hyp[1].strip().rstrip('<ANS>')
        new_hypotheses.append(scenario.replace('<BLK>', answer).strip())
    return new_hypotheses


    
def eval_bleu(split, prompt, out_type, references, out_file, method='corpus'):
    hypotheses = read_finetuned(out_file)
    hypotheses_basic = [ hyp.strip().split(':')[1].lstrip() for hyp in hypotheses] #

    
    if len(out_type)==0 and prompt=='all_tokens':
        # for tokens variants convent token tags to numbers
        hypotheses_modified = convert_tokens_to_basic_ungrouped(hypotheses_basic) #
        hypotheses_basic = hypotheses_modified
#     print(hypotheses_basic[0])
    
    n_references = []
    for refs in references:
        internal = []
        for r in refs:
            internal.append(r.split())
        # add same reference 5 times for 5 samples
        for i in range(NUM_SEQUENCES):
            n_references.append(internal)
#     print(n_references[0][0])
    scores= []
#     # one corpus
    if method == 'corpus':
        scores.append(corpus_bleu( n_references, [hyp.split() for hyp in hypotheses_basic], weights=(0.25,0.25,0.25,0.25), smoothing_function=chencherry.method1))
    elif method=='sent':
        for hypothesis, ref in zip(hypotheses_basic, n_references):
#         each sentence
            scores.append(sentence_bleu( ref, hypothesis.split(), weights=(0.25,0.25,0.25,0.25), smoothing_function=chencherry.method1))
    elif method=='scenario':
        for i in range(5):
            # each scenario
            scores.append(corpus_bleu( n_references[i*NUM_SEQUENCES: (i+1)*NUM_SEQUENCES], [hyp.split() for hyp in hypotheses_basic[i*NUM_SEQUENCES: (i+1)*NUM_SEQUENCES]], weights=(0.25,0.25,0.25,0.25), smoothing_function=chencherry.method1))
    print(np.mean(scores), np.std(scores))
    return scores

def eval_bleu_grouped(split, prompt, out_type, references, out_file):
    hypotheses = read_finetuned(out_file)
    hypotheses_basic = group_hypotheses(hypotheses) #[ hyp.strip().split(':')[1].lstrip() for hyp in hypotheses] #

    
    if len(out_type)==0 and prompt=='all_tokens':
        # for tokens variants convent token tags to numbers
        hypotheses_modified = convert_tokens_to_basic(hypotheses_basic) #convert_tokens_to_basic_ungrouped(hypotheses_basic) #
        hypotheses_basic = hypotheses_modified
#     print(hypotheses_basic[0])
    
    n_references = []
    for refs in references:
        internal = []
        for r in refs:
            internal.append(r.split())
        n_references.append(internal)
    
    scores= []
    for hypothesis, ref in zip(hypotheses_basic, n_references):
        scores.append(corpus_bleu( n_references, [hyp.split() for hyp in hypothesis], weights=(0.25,0.25,0.25,0.25), smoothing_function=chencherry.method1))
    print(np.mean(scores), np.std(scores))
    return scores

    
def read_k_fold_references(fold):
    with open('./data/folds/test_k_fold_references.json') as f:
        ref_dict = json.load(f)
    references = []
    for scenario in fold_dict[fold]:
        refs = ref_dict[scenario]
        refs = [ " ".join(r) for r in refs]
        references.append(refs)
    return references


def eval_bleu_for_k_fold(fold, split, prompt, out_type, method='corpus'):
    print(fold, split, prompt, out_type)
    references = read_k_fold_references(fold)
    out_file = './outputs/folds/generated_'+split+'_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1'+ out_type +'.txt'
    if method=='avg':
        scores = eval_bleu_grouped(split, prompt, out_type, references, out_file)
    else:
        scores = eval_bleu(split, prompt, out_type, references, out_file, method)
#     print(scores)

def eval_bleu_for_k_fold_manual(fold, split, prompt, out_type):
    print(fold, split, prompt, out_type)
    references = read_k_fold_references(fold)
    out_file = './outputs/eval/evaluation_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1'+ out_type +'.txt'
    scores = eval_bleu(split, prompt, out_type, references, out_file)
    
def eval_bleu_normal(split, prompt, out_type):
    print(split, prompt, out_type)
    references = read_references(file_path='./data/'+split+'_references.txt')
    out_file = './outputs/generated_'+split+'_'+prompt+'_large_g16_epoch1'+ out_type +'.txt'
    scores = eval_bleu(split, prompt, out_type, references, out_file)
    #print(scores)   
    
if __name__=="__main__":

    method = 'scenario' #'corpus' #'sent' #'avg' #
        
    for prompt in ['basic',  'ordered', 'direct', 'describe', 'expect', 'tokens', 'all_tokens']:#,
        for fold in range(1,9): 

            eval_bleu_for_k_fold(fold, 'test', prompt, '', method)
            eval_bleu_for_k_fold(fold, 'test', prompt, '_removed', method)
            eval_bleu_for_k_fold(fold, 'test', prompt, '_removed_deduplicated', method)
            eval_bleu_for_k_fold(fold, 'test', prompt, '_removed_deduplicated_ordered', method)
            
            # for manual evaluation data use below function in the same way
            # eval_bleu_for_k_fold_manual(fold, 'test', prompt, '_removed_deduplicated_ordered', method)

    


   
    
    

        
