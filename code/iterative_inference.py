import os, sys
import re
import numpy as np
from bs4 import BeautifulSoup
np.random.seed(42)


NUM_ITERATIONS = 10

def run_ilm(iteration):
    os.system("CUDA_VISIBLE_DEVICES=1 python ../transformers/code/run_generation.py --model_type=gpt2 --model_name_or_path=./models_gptl_ilm_num_ga8 --length=15 --k=50 --stop_token='<EOS>' --prompt=./outputs/iterative/input" + str(iteration)+ ".txt --output=./outputs/iterative-greedy/output"+ str(iteration)+".txt")
    
def read_output(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return lines

def replace_blanks(hypotheses):
    new_hypotheses = []
    for hyp in hypotheses:
        splitted_hyp = hyp.strip().split('<SEP>')
        scenario = splitted_hyp[0]
        answer = splitted_hyp[1].strip().rstrip('<ANS>')
        new_hypotheses.append(scenario.replace('<BLK>', answer).strip())
    return new_hypotheses
 
def iterative_mask_sentences_num(out_path, lines):
    with open(out_path, 'w') as o:
        for scenario in lines:
            splitted = scenario.split(":")
            scenario = splitted[1].rstrip('<EOS>')
            script = splitted[0].strip()
            new_scenario = script + ": "
            answer = "<SEP> "
            scenario = re.sub(r'\d+.', '</bevent> <bevent>', scenario)
            scenario = scenario + '</bevent>'
            scenario = scenario.strip().lstrip('</bevent>')
            soup = BeautifulSoup(scenario)
            events = []
            count=0
            for a in soup.find_all('bevent'):
                events.append(a.string)
            blk = np.random.randint(0,len(events))
            if len(events)>0:
                for idx, e in enumerate(events):
                    if e is not None:
                        if idx==blk:
                            new_scenario += str(idx+1) + ". <BLK> "
                            #answer +=  e.strip() + " <ANS> "
                        else:
                            new_scenario += str(idx+1) + ". " + e.strip() + " "
            o.write("{}\n".format(new_scenario + answer.strip()))
            
def create_next_input(iteration):
    hypotheses = read_output("./outputs/iterative-greedy/output"+ str(iteration)+".txt")
    new_input_hyps = replace_blanks(hypotheses)
    iterative_mask_sentences_num("./outputs/iterative-greedy/input" +str(iteration+1) + ".txt", new_input_hyps)
    
    
for it in range(NUM_ITERATIONS):
    run_ilm(it+1)
    create_next_input(it+1)