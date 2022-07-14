import dataclasses
import logging
import argparse
import os
import sys
import numpy as np
import json
import torch
from packaging import version
from datetime import datetime
import nltk


from nltk.corpus import stopwords 
from collections import defaultdict
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
from sklearn.metrics import f1_score


from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, BertForSequenceClassification, RobertaForSequenceClassification, AutoModelForSequenceClassification, BertConfig, XLNetForSequenceClassification, XLNetTokenizer

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.ERROR)
np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer),
                 "roberta": (RobertaForSequenceClassification, RobertaTokenizer),
                "xlnet": (XLNetForSequenceClassification, XLNetTokenizer)}

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class EventDataset(Dataset):
    
    def __init__(self, events, targets, tokenizer, max_len):
        self.events = events
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        event = str(self.events[item])
        target = self.targets[item]
        
        encoding = self.tokenizer.encode_plus(event, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, pad_to_max_length=True, truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
                
        
        return {
      'event_description': event,
      'input_ids': input_ids,
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def read_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        inputs = []
        labels = []
        for line in lines:
            inputs.append(line.strip()[0:-1].strip())
            labels.append(int(line.strip()[-1]))
    return inputs, labels    

        
def create_data_loader(events, targets, tokenizer, max_len, batch_size):
    
    ds = EventDataset(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
  )

def create_data_loader_test(events, targets, tokenizer, max_len, batch_size):
    
    ds = EventDataset(
    events=events,
    targets=targets,
    tokenizer=tokenizer,
    max_len=max_len
  )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4,
  )


def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler,
  n_examples
):
    losses = []
    f1score = []
    targets_all, preds_all = [], []
    correct_predictions = 0
    for d in data_loader:
        optimizer.zero_grad()
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
#         print(input_ids, attention_mask, targets, d['event_description'])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]
        #print(outputs.shape)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        
        correct_predictions += torch.sum(preds == targets)
        f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive classs
        losses.append(loss.item())
        targets_all += targets.cpu().numpy().tolist()
        preds_all += preds.cpu().numpy().tolist()
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        
        learning_rate = scheduler.get_last_lr()[0] if version.parse(torch.__version__) >= version.parse("1.4") else scheduler.get_lr()[0]
        
    return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all), learning_rate

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    f1score = []
    correct_predictions = 0
    
    targets_all, preds_all = [], [] 

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            f1score.append(f1_score(targets.cpu(), preds.cpu(), average='weighted')) # only for positive class
            losses.append(loss.item())
            targets_all += targets.cpu().numpy().tolist()
            preds_all += preds.cpu().numpy().tolist()
            
    return correct_predictions.double() / n_examples, np.mean(losses), f1_score(targets_all, preds_all)

    
def get_predictions(model, data_loader, output_dir, split):
    model = model.eval()
    event_descriptions = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with open(output_dir+'/'+split+'_tokens.txt', 'w') as f:
        with torch.no_grad():
            for d in data_loader:
                texts = d["event_description"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
#                 f.write("{}, {}, {}\n".format(texts, input_ids.cpu().detach().numpy(), targets.cpu().detach().numpy()))
                outputs = model(input_ids=input_ids,attention_mask=attention_mask)
                outputs = outputs[0]
                _, preds = torch.max(outputs, dim=1) #logits in first position of outputs

                probs = F.softmax(outputs, dim=1)
                event_descriptions.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        f1score = f1_score(real_values, predictions, average='weighted') # only for positive classs
    with open(output_dir+'/'+filename+'_predictions.txt', 'w') as f:
        for pred in predictions:
            f.write("{}\n".format(pred))
    with open(output_dir+'/'+ filename+ '_prediction_probs.txt', 'w') as f:
        for prob in prediction_probs:
            f.write("{}\n".format(prob.numpy()))
    return event_descriptions, predictions, prediction_probs, real_values


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Classifier", description="Fine-tunes a given Classifier")
    parser.add_argument('-train_data_path', '--train_data_path', required=True, help="path to train data file")
    parser.add_argument('-test_data_path', '--test_data_path', required=True, help="path to test data file")
    parser.add_argument('-val_data_path', '--val_data_path', required=True, help="path to val data file")
    parser.add_argument('-model_path_or_name', '--model_path', required=True, help="path to model")
    parser.add_argument('-filename', '--filename', default='dev', help="path to output file")
    parser.add_argument('-tokenizer', '--tokenizer', default="roberta-large", help="path to tokenizer")
    parser.add_argument('-model_type', '--model_type', required=True, help="type of model")
    parser.add_argument('-max_len', '--max_len', type=int, default=150, help="maximum length of the text")
    parser.add_argument('-batch_size', '--batch_size', type=int, default=8, help="size of each batch")
    parser.add_argument('-num_classes', '--num_classes', type=int, default=2, help="number of output classes")
    parser.add_argument('-num_train_epochs', '--epochs', type=int, default=20, help="number of trianing epochs")
#     parser.add_argument('-mask_prob', '--mask_prob', type=float, default=0.15, help="masking probability")
    parser.add_argument('--do_eval', action='store_true', help="to evaluate")
    parser.add_argument('--do_test', action='store_true', help="to test")
    parser.add_argument('--do_train',action='store_true', help="to train")
    parser.add_argument('-output_dir','--output_dir', type=str, default='./', help="output directory")
    parser.add_argument('--load', action='store_true', help="to load from trained-checkpoint")
#     parser.add_argument('--mask', action='store_true', help="to mask the inputs")
    parser.add_argument('--use_transformer', action='store_true', help="run a simple transformer model") 
    parser.add_argument('-trans_num_layer', '--trans_num_layer', type=int, default=1, help="number of layers in a simple transformer model")
    parser.add_argument('-dropout', '--dropout', type=float, default=0.1, help="dropout rate")
    parser.add_argument('-learning_rate', '--learning_rate', type=float, default=2e-5, help="learning rate")
    parser.add_argument('-decay', '--weight_decay', type=float, default=0.0, help="learning rate")
    
    args = parser.parse_args()
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path
    max_length = args.max_len
    batch_size = args.batch_size
    num_classes = args.num_classes
    model_type = args.model_type
    epochs = args.epochs
    do_train = args.do_train
    do_test = args.do_test
    output_dir = args.output_dir
    load_pretrained = args.load
    model_path = args.model_path
    learning_rate = args.learning_rate
    trans_num_layers = args.trans_num_layer
    dropout = args.dropout
    use_transformer = args.use_transformer
    do_eval = args.do_eval
    adam_epsilon = 1e-8
    warmup_steps = 0.06
    filename = args.filename
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    
    if args.tokenizer:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer)
        print("loading new tokenizer")
    elif args.model_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer"
        )


    
    if use_transformer:
        configuration = BertConfig()
        configuration.num_hidden_layers = trans_num_layers
        model = BertForSequenceClassification(configuration)
    else: 
        model = model_class.from_pretrained(model_path)

    model.config.hidden_dropout_prob = 0.1
    model.attention_probs_dropout_prob = dropout
    model.resize_token_embeddings(len(tokenizer))
    
    if do_test:
        model.load_state_dict(torch.load(output_dir+'/best_model_state.bin'))
    elif load_pretrained:
        model.load_state_dict(torch.load(output_dir+'/best_model_state_old.bin'))
        

     
    if do_train:
        train_events, train_targets = read_data(train_data_path)
   
        # load data
        train_data_loader = create_data_loader(train_events, train_targets, tokenizer, max_length, batch_size)
         
    if do_eval:
        val_events, val_targets = read_data(val_data_path)
        
        # load data
        val_data_loader = create_data_loader(val_events, val_targets, tokenizer, max_length, batch_size)
   

    model = model.to(device)
    
    # define loss function
    loss_fn = nn.CrossEntropyLoss().to(device)
 
    if do_train:
        start=datetime.now()
        # define optimizer, and scheduler
        total_steps = len(train_data_loader) * epochs
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps*0.06, num_training_steps=total_steps
        )


        # start training
        print("Training starts ....")
        history = defaultdict(list)
        best_score = 0

        model.zero_grad()
        model = model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            train_acc, train_loss, train_f1score, learning_rate = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_events))

            print(f'Train loss {train_loss} accuracy {train_acc} f1score {train_f1score} lr {learning_rate}')

            val_acc, val_loss, val_f1score = eval_model(model, val_data_loader, loss_fn,  device, len(val_events))
            print(f'Val   loss {val_loss} accuracy {val_acc} f1score {val_f1score}')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['train_f1score'].append(train_f1score)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            history['val_f1score'].append(val_f1score)

            if val_f1score > best_score:
                torch.save(model.state_dict(), output_dir+'/best_model_state.bin')
                best_score = val_f1score
        print("time taken to train", datetime.now()-start)
        
    if do_test: 
        test_events, test_targets = read_data(test_data_path)
#         train_events, train_targets = read_data(train_data_path)
#         val_events, val_targets = read_data(val_data_path)
        
        # load data
        test_data_loader = create_data_loader_test(test_events, test_targets, tokenizer, max_length, batch_size)
#         train_data_loader = create_data_loader_test(train_events, train_targets, tokenizer, max_length, batch_size)
#         val_data_loader = create_data_loader_test(val_events, val_targets, tokenizer, max_length, batch_size)

#         test_acc, _ , f1score = eval_model(model, test_data_loader, loss_fn, device, len(test_tweets))
        
        test_event_texts, test_pred, test_pred_probs, test_test = get_predictions(model, test_data_loader, output_dir, 'test')
#         train_event_texts, train_pred, train_pred_probs, train_test = get_predictions(model, train_data_loader, output_dir, 'train')
#         val_event_texts, val_pred, val_pred_probs, val_test = get_predictions(model, val_data_loader, output_dir, 'val')
        
        
#         with open(output_dir +'/test_events.txt', 'w') as f:
#             for t in test_event_texts:
#                 f.write("{}\n".format(t.strip()))
        print('--------------')
        print('filename: ', filename )
        print(test_event_texts[0:5])
        print('-----test report------')
        print(classification_report(test_test,test_pred))
        print(confusion_matrix(test_test, test_pred))
#         print(