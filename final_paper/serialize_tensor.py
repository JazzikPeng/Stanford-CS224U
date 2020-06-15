"""
Serialize torch tensor
"""
from train_semantic_probe import *

def serialize(dataset,
              encoder,
              path = "./serialized_data",
              featurizer = cls_featurizer,
              batch_size=10240,
            ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    # Construct PyTorch DataLoader
    dataloader = DataLoader(dataset, 
        shuffle=False,
        batch_size=1024,
        num_workers = 4)

    total_step = len(dataloader)
    encoder.to(device)

    # Serialize data
    print(f"Start Serializing: Number of data sample {total_step}")
    for step,  (X, X_mask, labels) in enumerate(tqdm(dataloader, desc="Serializing")):
        X = X.to(device)
        X_mask = X_mask.to(device)
        # BERT Encoder
        with torch.no_grad():
            output = encoder(X, attention_mask = X_mask)

        inputs = featurizer(output) # cls_token
        tensor_name = f"{featurizer.__name__}-{step}"
        inputs = inputs.detach().cpu().numpy()
        np.save(os.path.join(path, tensor_name), inputs)
    
if __name__ == "__main__":
    hf_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)
    bert_model = BertModel.from_pretrained(hf_weights_name)
    for param in bert_model.parameters():
        param.requires_grad = False
    train_dataset = PPDBDataset(corpus_path='./data/ppdb_train',
                        tokenizer=bert_tokenizer,
                        encoder=bert_model,
                        seq_len=128)

    serialize(train_dataset, encoder=bert_model)


