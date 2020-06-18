import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import TweetTokenizer
from writeprints import get_writeprints_transformer, prepare_entry
from tqdm import  tqdm
import json
import re
import os
import string
import argparse
import sys
import torch

TRANSFORMER_FILE = 'transformers.p'
MODEL_FILE = 'best_model.pt'

class PANDatasetIterator(torch.utils.data.IterableDataset):

    def __init__(self, f_in, transformer, scaler):
        self.f_in = f_in
        self.transformer = transformer
        self.scaler = scaler

    def mapper(self, line):
        d = json.loads(line)
        x1 = scaler.transform(transformer.transform([prepare_entry(d['pair'][0])]).todense())
        x2 = scaler.transform(transformer.transform([prepare_entry(d['pair'][1])]).todense())
        x = np.abs(x1 - x2)[0, :].astype('float32')
        return x, d['id']
    

    def __iter__(self):
        f_itr = open(self.f_in, 'r')
        return map(self.mapper, f_itr)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction Script: PAN 2020 - Janith Weerasinghe')
    parser.add_argument('-i', type=str,
                        help='Evaluaiton dir')
    parser.add_argument('-o', type=str, 
                        help='Output dir')
    args = parser.parse_args()
    
    # validate:
    if not args.i:
        raise ValueError('Eval dir path is required')
    if not args.o:
        raise ValueError('Output dir path is required')
        
        
    input_file = os.path.join(args.i, 'pairs.jsonl')
    output_file = os.path.join(args.o, 'answers.jsonl')
    print("Writing answers to:", output_file , file=sys.stderr)
    
    with open(TRANSFORMER_FILE, 'rb') as f:
        transformer, scaler = pickle.load(f)

    with open(MODEL_FILE, 'rb') as f:
        best_model = torch.load(f)
    
    device = torch.device('cpu')
    ds = PANDatasetIterator(input_file, transformer, scaler)
    test_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=1000)

    fout = open(output_file, 'a')
    c = 0
    with torch.no_grad():
        for x, ids in test_loader:
            x = x.to(device)
            outputs = best_model(x)
            probs = outputs.numpy()[:, 0].astype(float)

            for i in range(len(ids)):
                d = {
                    'id': ids[i],
                    'value': probs[i]
                }
                json.dump(d, fout)
                fout.write('\n')
            c += len(ids)
            print(c, file=sys.stderr)
            print('Written to', output_file, flush=True, file=sys.stderr)

    fout.close()
    print("Execution complete", file=sys.stderr)
                