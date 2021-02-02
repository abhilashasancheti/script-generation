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
from itertools import combinations

np.random.seed(42)
random.seed(42)
NUM_SEQUENCES=5

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


def test_classification_all_combinations_partial_data_from_lines(lines, out_path):
    # input is references from test_token_references with test.txt
    with open(out_path, 'w') as o:
        for scenario in lines:
            splitted = scenario.strip(' ').split(": ")
            scenario = splitted[1].strip()
            scenario = scenario.replace('<EEVENT>','</bevent>').replace('<BEVENT>','<bevent>')
            script = splitted[0].rstrip(' <ESCR>').replace("<BOS> <SCR> ","")
            new_scenario = script + ": "
            answer = "<SEP> "
            soup = BeautifulSoup(scenario)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            
            all_combinations = combinations(range(len(events)), 2)
            if len(events)>0:
                for eids in all_combinations:
                    new_scenario = script.strip() + ": "
                    answer = ""
                    label = np.random.randint(0,2)
                    first=True
                    advance_event = ""
                    for idx, e in enumerate(events):
                        if idx in list(eids):
                            if label==1 and first:
                                answer +=  "</s> " + e.strip() + " "
                            elif label==0 and first:
                                advance_event = "</s> " + e.strip() + " "
                            else:
                                answer += "</s> " + e.strip() + " "
                            first = False
                    answer += advance_event
                    o.write("{}\n".format(new_scenario + answer + str(label)))
                    
def train_relevant_classification_positive_data(in_path, out_path):
    # input is all tokens file
    with open(in_path) as f, open(out_path, 'w') as o:
        lines = f.readlines()
        scene_dict = {}
        for scenario in lines:
            splitted = scenario.split(": ")
            scenario = splitted[1].strip().rstrip('<EOS>')
            script = splitted[0].strip().lstrip('<BOS> <SCR> ').rstrip('<ESCR>')
            scenario = scenario.replace('<EEVENT>','</bevent>').replace('<BEVENT>','<bevent>')
            soup = BeautifulSoup(scenario)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            label = 1 #only positive examples now
            if script.strip() not in scene_dict:
                scene_dict[script.strip()] = []
            if len(events)>0:
                for idx, e in enumerate(events):
                    new_scenario = script.strip() + ": "
                    answer = ""
                    scene_dict[script.strip()].append(e.strip())
                    answer +=  "</s> " + e.strip() + " "
                    o.write("{}\n".format(new_scenario + answer + str(label)))
    return scene_dict

def generated_relevant_classification_positive_data(in_path, out_path, prompt):
    # input is generated output in numbered form
    with open(in_path) as f, open(out_path, 'w') as o:
        lines = f.readlines()
        for scenario in lines:
            splitted = scenario.split(": ")
            scenario = splitted[1].rstrip(' <EOS>')
#             print(scenario)
            if prompt=='direct':
                script = splitted[0].strip().lstrip('<BOS> <SCR> ').rstrip('<ESCR>') # direct
            elif prompt=='describe':
                script = splitted[0].strip().replace("<BOS> describe ","") # for describe
                script = script.replace(" in small sequences of short sentences","") #for describe 
            elif prompt=='expect':
                script = splitted[0].strip().replace("<BOS> these are the things that happen when you ","") # expect
                script = dict_script[script]
            elif prompt=='ordered':
                script = splitted[0].strip().replace("<BOS> here is an ordered sequence of events that occur when you ","")
                script = dict_script[script]
            elif prompt=='basic':
                script = splitted[0].strip().replace("<BOS> here is a sequence of events that happen while ","")
            else:
                script = splitted[0].rstrip(' <ESCR>').replace("<BOS> <SCR> ","")
            new_scenario = script + ": "
            if scenario.startswith('<BEVENT>'):
                scenario = re.sub(r'<EEVENT>', '</bevent>', scenario)
            else:
                scenario = re.sub(r'\d+[.]', '</bevent> <bevent>', scenario)
                scenario = scenario.strip() + ' </bevent>'
                scenario = scenario.strip().lstrip('</bevent>')
            soup = BeautifulSoup(scenario)
#             print(soup)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            label = 1 #only positive examples now

            if len(events)>0:
                for idx, e in enumerate(events):
                    new_scenario = script.strip() + ": "
                    answer = ""
                    answer +=  "</s> " + e.strip() + " "
#                     print(new_scenario + answer + str(label))
                    o.write("{}\n".format(new_scenario + answer + str(label)))

def evaluation_relevant_classification_positive_data(in_path, out_path):
    # input is generated output in numbered form
    with open(in_path) as f, open(out_path, 'w') as o:
        lines = f.readlines()
        for scenario in lines:
            splitted = scenario.split(": ")
            scenario = splitted[1].rstrip(' <EOS>')
            #prompt = "here is an ordered sequence of events that occur when you"
            if prompt=='direct':
                script = splitted[0].strip().lstrip('<BOS> <SCR> ').rstrip('<ESCR>') # direct
            elif prompt=='describe':
                script = splitted[0].strip().replace("<BOS> describe ","") # for describe
                script = script.replace(" in small sequences of short sentences","") #for describe 
            elif prompt=='expect':
                script = splitted[0].strip().replace("<BOS> these are the things that happen when you ","") # expect
                script = dict_script[script]
            elif prompt=='ordered':
                script = splitted[0].strip().replace("<BOS> here is an ordered sequence of events that occur when you ","")
                script = dict_script[script]
            elif prompt=='basic':
                script = splitted[0].strip().replace("<BOS> here is a sequence of events that happen while ","")
            else:
                script = splitted[0].rstrip(' <ESCR>').replace("<BOS> <SCR> ","")
            new_scenario = script + ": "
            scenario = re.sub(r'\d+[.]', '</bevent> <bevent>', scenario)
            scenario = re.sub(r'<EEVENT>', '</bevent>', scenario)
            scenario = scenario.strip() + ' </bevent>'
            scenario = scenario.strip().lstrip('</bevent>')
            soup = BeautifulSoup(scenario)
#             print(soup)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)

            if len(events)>0:
                for idx, e in enumerate(events):
                    new_scenario = script.strip() + ";"
                    answer = ""
                    answer +=  e.strip() + " "
                    o.write("{}\n".format(new_scenario + answer))

def test_relevant_classification_positive_data(lines, out_path):
    # input is tokens references with text.txt
    with open(out_path, 'w') as o:
        for scenario in lines:
            splitted = scenario.split(": ")
            scenario = splitted[1].strip()
            script = splitted[0].strip().lstrip('<BOS> <SCR> ').rstrip('<ESCR>')
            scenario = scenario.replace('<EEVENT>','</bevent>').replace('<BEVENT>','<bevent>')
            soup = BeautifulSoup(scenario)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            label = 1 #only positive examples now

            if len(events)>0:
                for idx, e in enumerate(events):
                    new_scenario = script.strip() + ": "
                    answer = ""
                    answer +=  "</s> " + e.strip() + " "
                    o.write("{}\n".format(new_scenario + answer + str(label)))



def train_relevant_classification_negative_data(scene_dict, out_path):
    with open(out_path, 'a') as f:
        scenes = list(scene_dict.keys())
        label = 0 # negative example
        for idx, scene in enumerate(scenes):
            for i in range(len(scene_dict[scene])):
                temp_scenes = copy.deepcopy(scenes)
                temp_scenes.remove(scene) # remove the scence under consideration 
                contrastive_scene = np.random.choice(temp_scenes, 1)
                event = np.random.choice(scene_dict[contrastive_scene[0]],1) # only onne element in the list
                new_scenario = scene.strip() + ": "
                answer = ""
                answer +=  "</s> " + event[0].strip() + " "
                f.write("{}\n".format(new_scenario + answer + str(label)))

def classification_all_combinations_partial_data(in_path, out_path):
    with open(in_path) as f, open(out_path, 'w') as o:
        lines = f.readlines()
        for scenario in lines:
            splitted = scenario.split(": ")
            scenario = splitted[1].rstrip('<EOS>')
            script = splitted[0].strip().lstrip('<BOS> <SCR> ').rstrip('<ESCR>')
            scenario = scenario.replace('<EEVENT>','</bevent>').replace('<BEVENT>','<bevent>')
            soup = BeautifulSoup(scenario)
            #print(soup, scenario)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            
            all_combinations = combinations(range(len(events)), 2)
            if len(events)>0:
                for eids in all_combinations:
                    new_scenario = script.strip() + ": "
                    answer = ""
                    label = np.random.randint(0,2)
                    first=True
                    advance_event = ""
                    for idx, e in enumerate(events):
                        if idx in list(eids):
                            if label==1 and first:
                                answer +=  "</s> " + e.strip() + " "
                            elif label==0 and first:
                                advance_event = "</s> " + e.strip() + " "
                            else:
                                answer += "</s> " + e.strip() + " "
                            first = False
                    answer += advance_event
                    o.write("{}\n".format(new_scenario + answer + str(label)))
                    
def test_classification_all_combinations_partial_data(in_path, out_path, prompt):
    # input is generated output in numbered form
    with open(in_path) as f, open(out_path, 'w') as o:
        lines = f.readlines()
        for scenario in lines:
            splitted = scenario.split(": ")
            scenario = splitted[1].rstrip(' <EOS>')
            if prompt=='direct':
                script = splitted[0].strip().lstrip('<BOS> <SCR> ').rstrip('<ESCR>') # direct
            elif prompt=='describe':
                script = splitted[0].strip().replace("<BOS> describe ","") # for describe
                script = script.replace(" in small sequences of short sentences","") #for describe 
            elif prompt=='expect':
                script = splitted[0].strip().replace("<BOS> these are the things that happen when you ","") # expect
                script = dict_script[script]
            elif prompt=='ordered':
                script = splitted[0].strip().replace("<BOS> here is an ordered sequence of events that occur when you ","")
                script = dict_script[script]
            elif prompt=='basic':
                script = splitted[0].strip().replace("<BOS> here is a sequence of events that happen while ","")
            else:
                script = splitted[0].rstrip(' <ESCR>').replace("<BOS> <SCR> ","")
            new_scenario = script + ": "
            answer = "<SEP> "
            scenario = re.sub(r'\d+[.]', '</bevent> <bevent>', scenario)
#             scenario = re.sub(r'<EEVENT>', '</bevent>', scenario)
            scenario = scenario + '</bevent>'
            scenario = scenario.strip().lstrip('</bevent>')
            soup = BeautifulSoup(scenario)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            
            all_combinations = combinations(range(len(events)), 2)
            if len(events)>0:
                for eids in all_combinations:
                    new_scenario = script.strip() + ": "
                    answer = ""
                    label = np.random.randint(0,2)
                    first=True
                    advance_event = ""
                    for idx, e in enumerate(events):
                        if idx in list(eids):
                            if label==1 and first:
                                answer +=  "</s> " + e.strip() + " "
                            elif label==0 and first:
                                advance_event = "</s> " + e.strip() + " "
                            else:
                                answer += "</s> " + e.strip() + " "
                            first = False
                    answer += advance_event
                    o.write("{}\n".format(new_scenario + answer + str(label)))

def evaluation_data(in_path, out_order_path, out_rel_path, out_path, prompt, percent=0.60):
    # input is generated output in numbered form
    with open(in_path) as f, open(out_path, 'w') as s, open(out_order_path, 'w') as o, open(out_rel_path, 'w') as g:
        lines = f.readlines()
        for l in range(0, len(lines),5): #scenario in lines:#
            # sample one sequence
            ind = np.random.choice(NUM_SEQUENCES, 1, replace=False)
            scenario = lines[l + ind[l//5]]
            s.write("{}\n".format(scenario.strip()))
            splitted = scenario.split(": ")
            scenario = splitted[1].rstrip(' <EOS>')
            if prompt=='direct':
                script = splitted[0].strip().lstrip('<BOS> <SCR> ').rstrip('<ESCR>') # direct
            elif prompt=='describe':
                script = splitted[0].strip().replace("<BOS> describe ","") # for describe
                script = script.replace(" in small sequences of short sentences","") #for describe 
            elif prompt=='expect':
                script = splitted[0].strip().replace("<BOS> these are the things that happen when you ","") # expect
                script = dict_script[script]
            elif prompt=='ordered':
                script = splitted[0].strip().replace("<BOS> here is an ordered sequence of events that occur when you ","")
                script = dict_script[script]
            elif prompt=='basic':
                script = splitted[0].strip().replace("<BOS> here is a sequence of events that happen while ","")
            else:
                script = splitted[0].rstrip(' <ESCR>').replace("<BOS> <SCR> ","")

            scenario = re.sub(r'\d+[.]', '</bevent> <bevent>', scenario)
            scenario = re.sub(r'<EEVENT>', '</bevent>', scenario)
            scenario = scenario + '</bevent>'
            scenario = scenario.strip().lstrip('</bevent>')
            soup = BeautifulSoup(scenario)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            
            ids = np.random.choice(len(events)-1, int(percent*(len(events)-1)), replace=False)
            
            if len(events)>0:
                for i in ids:
                    new_scenario = script.strip() + ";"
                    answer = events[i].strip() + ";" + events[i+1].strip()
                    o.write("{}\n".format(new_scenario + answer))
            
            # sample events for relevancy
            sampled_events = np.random.choice(len(events), int(percent*(len(events))), replace=False)
            if len(events)>0:
                for idx in sampled_events:
                    new_scenario = script.strip() + ";"
                    answer = ""
                    answer +=  events[idx].strip() + " "
                    g.write("{}\n".format(new_scenario + answer))

def read_k_fold_references(fold):
    with open('./data/folds/test_k_fold_tokens_references.json') as f:
        ref_dict = json.load(f)
    references = []
    for scenario in fold_dict[fold]:
        refs = ref_dict[scenario]
        refs = [ " ".join(r) for r in refs]
        references.append(refs)
    return references
                    
def create_lines_for_gt_classification(fold):
    references = read_k_fold_references(fold)
    with open("./data/folds/test_tokens_fold"+str(fold)+".txt") as f:
        scenarios = f.readlines()
    lines = []
    for idx, ref in enumerate(references):
        for r in ref:
            lines.append(scenarios[idx].strip() + " " + r )
    return lines        


    


if __name__=="__main__":
#     ordering train data
#     for fold in range(1,9):
#         classification_all_combinations_partial_data("./data/folds/train_all_tokens_fold"+str(fold)+"_shuf.txt", "./data/folds/train_classification_partial_context_all_fold"+str(fold)+".txt")
#         lines = create_lines_for_gt_classification(fold)
#         test_classification_all_combinations_partial_data_from_lines(lines,"./data/folds/test_classification_partial_context_all_fold"+str(fold)+".txt")
    
    for prompt in ['basic', 'ordered', 'direct', 'describe', 'expect', 'tokens', 'all_tokens']:
        for fold in range(1,9):
            test_classification_all_combinations_partial_data("./outputs/folds/generated_test_" +prompt+ "_large_fold"+str(fold)+"_g16_epoch1_removed_deduplicated.txt","./data/folds/test_classification_"+prompt+"_fold"+str(fold)+"_output_removed_deduplicated.txt", prompt)

#     relevancy train data
#     for fold in range(1,9):
#         scene_dict = train_relevant_classification_positive_data("./data/folds/train_all_tokens_fold"+str(fold)+"_shuf.txt", './data/folds/train_relevant_classification_fold'+str(fold)+'.txt')
#         train_relevant_classification_negative_data(scene_dict, './data/folds/train_relevant_classification_fold'+str(fold)+'.txt')
#         lines = create_lines_for_gt_classification(fold)
#         test_relevant_classification_positive_data(lines, './data/folds/test_relevant_classification_fold'+str(fold)+'.txt')
#         for prompt in ['basic', 'expect', 'ordered', 'direct', 'describe', 'tokens', 'all_tokens']:
#             generated_relevant_classification_positive_data('./outputs/folds/generated_test_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1.txt', './data/folds/test_relevant_classification_'+prompt+'_fold'+str(fold)+'_output.txt', prompt)
#         manual evaluation data for fold data
#         for fold in [4,7]:
#             in_path = './outputs/folds/generated_test_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             out_order_path = './outputs/eval/evaluation_ordering_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             out_rel_path = './outputs/eval/evaluation_relevancy_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             out_path = './outputs/eval/evaluation_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             evaluation_data(in_path, out_order_path, out_rel_path, out_path, prompt, percent=1.0)
    
#         manual evaluation data for novel scenarios
#     for prompt in ['basic']:
#         for fold in [1]:
#             in_path = './outputs/folds/generated_new_scenarios_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             out_order_path = './outputs/eval_novel/evaluation_ordering_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             out_rel_path = './outputs/eval_novel/evaluation_relevancy_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             out_path = './outputs/eval_novel/evaluation_'+prompt+'_large_fold'+str(fold)+'_g16_epoch1_removed_deduplicated_ordered.txt'
#             evaluation_data(in_path, out_order_path, out_rel_path, out_path, prompt, percent=1.0)
    