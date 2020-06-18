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

TRANSFORMER_FILE = 'transformers.p'
MODEL_FILE = 'LiniearRegressionModal.p'

def process_batch(transformer, scaler, clf, ids, preprocessed_docs1, preprocessed_docs2, output_file):
    print('Extracting features:', len(ids), file=sys.stderr)
    X1 = scaler.transform(transformer.transform(preprocessed_docs1).todense())
    X2 = scaler.transform(transformer.transform(preprocessed_docs2).todense())
    X = np.abs(X1 - X2)
    print('Predicting...', file=sys.stderr)
    probs = clf.predict_proba(X)[:, 1]
    print('Writing to', output_file, file=sys.stderr)
    with open(output_file, 'a') as f:
        for i in range(len(ids)):
            d = {
                'id': ids[i],
                'value': probs[i]
            }
            json.dump(d, f)
            f.write('\n')
            
    
    
    
    
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
        clf = pickle.load(f)
    
    preprocessed_docs1 = []
    preprocessed_docs2 = []
    ids = []
    batch_size = 100
    with open(input_file, 'r') as f:
        i = 0
        for l in f:
            if i % 100 == 0:
                print(i, file=sys.stderr)
            i += 1
            
            d = json.loads(l)
            ids.append(d['id'])
            preprocessed_docs1.append(prepare_entry(d['pair'][0]))
            preprocessed_docs2.append(prepare_entry(d['pair'][1]))
            if len(ids) >= batch_size:
                process_batch(transformer, scaler, clf, ids, preprocessed_docs1, preprocessed_docs2, output_file)
                preprocessed_docs1 = []
                preprocessed_docs2 = []
                ids = []
        process_batch(transformer, scaler, clf, ids, preprocessed_docs1, preprocessed_docs2, output_file)
    print("Execution complete", file=sys.stderr)
                