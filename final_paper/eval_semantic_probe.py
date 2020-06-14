"""
Read model from model_checkpoint and compute f1 score
"""
import numpy as np
import random
from transformers import BertModel, BertTokenizer
from classifier import logisticRegressionClassifier
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange


import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from train_semantic_probe import PPDBDataset, cls_featurizer

def test(dataloader, classifier, encoder, device):
    classifier.eval()
    y_true, y_pred = [], []
    for step,  (X, X_mask, labels) in enumerate(tqdm(dataloader, desc="Iteration")):
        X = X.to(device)
        X_mask = X_mask.to(device)
        # BERT Encoder
        output = encoder(X, attention_mask = X_mask)
        inputs = cls_featurizer(output) # cls_token
        inputs = inputs.to(device)
        outputs = classifier(inputs)
        y_true.extend(list(labels.numpy()))
        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        # print("load pred", pred)
        y_pred.extend(list(pred))
    classifier.train()
    # Compute F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return f1, acc

if __name__ == "__main__":
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    hf_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)
    bert_model = BertModel.from_pretrained(hf_weights_name)
    for param in bert_model.parameters():
        param.requires_grad = False
        
    dataset = PPDBDataset(corpus_path='./data/ppdb_test',
                        tokenizer=bert_tokenizer,
                        encoder=bert_model,
                        seq_len=128)
                        
    eval_dataloader = DataLoader(dataset, 
        shuffle = False,
        batch_size=512,
        num_workers = 4)

    # Read trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "./model_checkpoint/logisticRegressionClassifier-4"

    classifier = logisticRegressionClassifier(2, input_dim=768)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    
    f1, acc = test(eval_dataloader, classifier, encoder=bert_model, device = device)
    print(f"f1 : {f1}| acc: {acc}")
