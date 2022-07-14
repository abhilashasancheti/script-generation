# script-generation
This repository contains code and data for STARSEM2022 paper titled "What do Large Language Models Learn about Scripts?"


## Steps for the proposed pipeline
1. Fine tune LM, train relevance, and temporal classifier as mentioned below.
2. Generate scripts from LM as mentioned below.
3. Using the generated output, create test data for relevance classifier using `prepare_classification_data.py`
4. Get predictions from relevance classifier using command mentioned below.
5. Remove irrelevant events and deduplicate using commands in cell 61 of `post_process_outputs.ipynb`. This results in generation of output files with relevant and deduplicated events.
6. Create test data for temporal ordering classifer using `prepare_classification_data.py`.
7. Run topological sort using cell 65 in `post_process_outputs.ipynb` to get the final output files.

### A one-step post-processing pipeline is coming soon..

## Fine-tuning the LM
```
mkdir models
mkdir logging
mkdir train_outputs
cd code.py
```

### GPT2
```
CUDA_VISIBLE_DEVICES=0 python run_language_modeling.py --output_dir=../models/models_gptl_basic_fold6_g16 --model_type=gpt2 --model_name_or_path=gpt2-large --do_train --train_data_file=../data/folds/train_basic_fold6_shuf.txt --tokenizer=../gpt-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --per_device_eval_batch_size=1 --per_device_train_batch_size=1 --block_size=150  --logging_steps=100 --logging_dir=../logging/ --line_by_line --gradient_accumulation_steps=16 > ../train_outputs/folds/lm_large_epoch1_basic_fold6_g16.txt
```

### BART
```
CUDA_VISIBLE_DEVICES=0 python run_seq2seq_bart.py --model_name_or_path facebook/bart-large --do_train --task summarization --train_file ../data/folds/seq2seq/train_ordered_fold1.csv --output_dir ../models_bartl_ordered_fold1_g16 --max_source_length 15 --max_target_length 150 --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --text_column input --summary_column output --tokenizer=../bart-tokenizer --num_train_epochs=1 --overwrite_output_dir --save_steps=100 --logging_steps=100 --logging_dir=../logging/ --gradient_accumulation_steps=16 > ../train_outputs/folds-bart/bart_large_epoch1_ordered_fold1_g16.txt
```
### T5
```
CUDA_VISIBLE_DEVICES=0 python run_seq2seq.py --model_name_or_path t5-large --do_train --task summarization --train_file ../data/folds/t5/train_ordered_fold1.csv --output_dir ../models_t5l_ordered_fold1_g16 --max_source_length 15 --max_target_length 150 --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --text_column input --summary_column output --tokenizer=../t5-tokenizer --num_train_epochs=3 --overwrite_output_dir --save_steps=100 --logging_steps=100 --logging_dir=../logging/ --gradient_accumulation_steps=16 > ../train_outputs/folds-t5/t5_large_epoch3_ordered_fold1_g16.txt
```

## Generating from the LM
```
mkdir -p outputs/folds
cd code.py
```

### GPT-2
```
CUDA_VISIBLE_DEVICES=0 python run_generation.py --model_type=gpt2 --model_name_or_path=../models_gptl_ordered_fold8_g16 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>' --prompt=../data/folds/test_ordered_fold8.txt --output ../outputs/folds/generated_test_ordered_large_fold8_g16_epoch1.txt
```    
### BART
```
CUDA_VISIBLE_DEVICES=0 python run_eval_bart.py ../models_bartl_basic_fold4_g16 ../bart-tokenizer ../data/folds/gpt2/test_basic_fold4.txt ../outputs/folds/bart/generated_test_basic_large_fold4_g16_epoch1.txt --reference_path ../data/folds/gpt2/test_basic_fold4.txt --score_path enro_bleu.json --task summarization --n_obs 100 --device cuda --bs 5 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>'
``` 
  
### T5
```
CUDA_VISIBLE_DEVICES=0 python run_eval.py ../models_t5l_expect_fold1_g16 ../t5-tokenizer ../data/folds/t5/test_expect_fold1.txt ../outputs/folds/t5/generated_test_expect_large_fold1_g16_epoch3.txt --reference_path ../data/folds/t5/test_expect_fold1.txt --score_path enro_bleu.json --task summarization --n_obs 100 --device cuda --bs 5 --length=150 --k=50 --num_return_sequences=5 --sample --stop_token='<EOS>'
```   
    
## Relvance Classifier 
For training the relavance classifier, first create data using the below commands. modify the paths accordingly in the file and comment the non-relevant snippets from the main function. For the dataset used in the paper, we provide the trianing data under ```data/folds/``` folder.
``` 
cd code
python prepare_classification_data.py
```

### Training the relevance classifier
To train the classifier for each fold use the below command by replacing the fold numbers as needed.

```
cd code
CUDA_VISIBLE_DEVICES=0 python run_classification.py -train_data_path ../data/folds/train_relevant_classification_fold7.txt -val_data_path ../data/folds/valid_relevant_classification_fold7.txt  -test_data_path ../data/folds/test_relevant_classification_fold7.txt  -model_path_or_name=roberta-large -model_type roberta --do_train --do_eval -output_dir ../folds/roberta-relevant-classification-fold7-e5 -tokenizer ../roberta-tokenizer/ -num_train_epochs 5 -batch_size 16 > ../train_outputs/folds/relevant_classifcation_fold7_e5.txt
```

### Testing the relevance classifier
```
cd code
CUDA_VISIBLE_DEVICES=0 python run_classification.py -train_data_path ../data/folds/train_relevant_classification_fold7.txt -val_data_path ../data/folds/valid_relevant_classification_fold7.txt  -test_data_path ../data/folds/test_relevant_classification_fold7.txt  -model_path_or_name=roberta-large -model_type roberta --do_test -output_dir ../folds/roberta-relevant-classification-fold7-e5 -tokenizer ../roberta-tokenizer/ -batch_size 16 --filename fold7
```

## Temporal Ordering Classifier 
For training the temporal ordering classifier, first create data using the below commands. modify the paths accordingly in the file and comment the non-relevant snippets from the main function. For the dataset used in the paper, we provide the trianing data under ```data/folds/``` folder.
```
cd code
python prepare_classification_data.py
```

### Training the temporal ordering classifier
To train the classifier for each fold use the below command by replacing the fold numbers as needed.

```
cd code
CUDA_VISIBLE_DEVICES=0 python run_classification.py -train_data_path ../data/folds/train_classification_partial_context_all_fold4.txt -val_data_path ../data/folds/valid_classification_partial_context_all_fold4.txt  -test_data_path ../data/folds/test_classification_partial_context_all_fold4.txt  -model_path_or_name=roberta-large -model_type roberta --do_train --do_eval -output_dir ./folds/roberta-classification-fold4-e10 -tokenizer ../roberta-tokenizer/ -num_train_epochs 10 -batch_size 16 > ../train_outputs/folds/classifcation_fold4_e10.txt
```

### Testing the temporal ordering classifier
```
cd code
CUDA_VISIBLE_DEVICES=0 python run_classification.py -train_data_path ../data/folds/train_classification_partial_context_all_fold3.txt -val_data_path ../data/folds/valid_classification_partial_context_all_fold3.txt  -test_data_path ../data/folds/valid_classification_partial_context_all_fold3.txt  -model_path_or_name=roberta-large -model_type roberta --do_test -output_dir ../folds/roberta-classification-fold3-e10 -tokenizer ../roberta-tokenizer/ -batch_size 16 --filename valid_fold3
```
    
## For adding tokens to a tokenizer

```
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-large')
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'sep_token': '<SEP>', 'additional_special_tokens': ['<ANS>','<BLK>','<SCR>','<ESCR>', '<BEVENT>', '<EEVENT>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.save_pretrained('./t5-tokenizer/')

from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'sep_token': '<SEP>', 'additional_special_tokens': ['<ANS>','<BLK>','<SCR>','<ESCR>', '<BEVENT>', '<EEVENT>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.save_pretrained('./bart-tokenizer/')
 ```
 
 
