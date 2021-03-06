"""
Serialize torch tensor
"""
from train_semantic_probe import *

def serialize(dataset,
              encoder,
              featurizer,
              path = "./serialized_data",
              batch_size=25600,
              bert_layer=-1
            ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for serializing")
    # Construct PyTorch DataLoader
    dataloader = DataLoader(dataset, 
        shuffle=False,
        batch_size=batch_size,
        num_workers = 4)

    total_step = len(dataset)
    encoder.to(device)

    # Serialize data
    print(f"Start Serializing: Number of data sample {total_step}")
    for step,  (X, X_mask, labels) in enumerate(tqdm(dataloader, desc="Serializing")):
        X = X.to(device)
        X_mask = X_mask.to(device)
        # BERT Encoder
        with torch.no_grad():
            final_hidden_states, cls_output, output_hidden_states = encoder(X, attention_mask = X_mask)
        output = output_hidden_states[bert_layer]
        inputs = featurizer(output) 
        tensor_name = f"{featurizer.__name__}-{step}"
        inputs = inputs.detach().cpu().numpy()
        sample = np.array([inputs, labels]) 
        np.save(os.path.join(path, tensor_name), sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input ppdb pairs.")

    parser.add_argument("--featurizer",
                        default=None,
                        type=str,
                        required=True,
                        help="Featurizer used applied on BERT output")
    
    parser.add_argument("--output_dir",
                    type=str,
                    required=True,
                    help="Directory to save serialized data")

    parser.add_argument("--tensor_size",
                default = 10240,
                type=int,
                required=False,
                help="Number of sample per array saved")
    
    parser.add_argument("--bert_layer",
                default=-1, # Last layer out put 
                type=int,
                required=False,
                help="Extract certain layer of BERT output, 0 to 12")
    
    args = parser.parse_args()

    if args.featurizer == "cls_featurizer":
        feat = featurizer.cls_featurizer
    elif args.featurizer == "avg_pooling_featurizer":
        feat = featurizer.avg_pooling_featurizer
    else:
        raise ValueError("Please enter name of existing featurizer")
    
    assert args.bert_layer >=0 and args.bert_layer <= 12, "Incorrect bert layer value, 0 to"

    create_directory(args.output_dir)

    hf_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)
    if args.bert_layer == -1:
        bert_model = BertModel.from_pretrained(hf_weights_name)
    else:
        bert_model = BertModel.from_pretrained(hf_weights_name, output_hidden_states=True)

    for param in bert_model.parameters():
        param.requires_grad = False
    train_dataset = PPDBDataset(corpus_path=args.data_path,
                        tokenizer=bert_tokenizer,
                        encoder=bert_model,
                        seq_len=128)

    serialize(train_dataset, encoder=bert_model, 
        featurizer=feat, 
        path=args.output_dir, 
        batch_size=args.tensor_size, 
        bert_layer=args.bert_layer)


