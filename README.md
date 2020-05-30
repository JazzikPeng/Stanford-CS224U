# Problem Statement
Sperate semantic meaning from language representation models (e.g. BERT). This is a more fundamental topic compares to the first one. Here is the hypothesis. If we use BERT to encode a sentence, the vector will contain both semantic information and syntactic information. Here is one example. 

    Target: I love to play video games.

    Candidate 1: Gaming is my favorite hobby.

    Candidate 2: I love to eat apples.

Which candidate is more similar to Target? Semantically, candidate 1 clearly wins, but syntactically, candidate 2 might be closer. This can be a very difficult question for language representation models like BERT. We want to train a model to find a sub-vector that contains only semantic information. There are various applications for this, such as redundancy detection. 

# Setup 
1. Download [ppdb](http://paraphrase.org/#/download) data. Please download English and S Size for running it locally. Save downloaded data to the `data` directory and unzip it.

2. Activate nlu virtual environment and Run prepro_ppdb.py. This file does some basic data preprocessing. 
```bash
conda activate nlu

python3 prepro_ppdb.py --in_file_path ./data/ppdb-2.0-s-all --out_file_path ./data/ppdb_train --orig_in_format
```

3. Run finetune_bert_with_ppdb.py. This file fine tune the BERT model on ppdb dataset.
   
```bash
python3 finetune_bert_with_ppdb.py --train_corpus ./data/ppdb_train --bert_model bert-base-uncased --output_dir checkpoint --do_train
```
The model will start tunning. It will take long time without an GPU.

# EC2 Instance Setup
Use **p2.xlarge** GPU instance with K80 GPU. Some instance uses Kepler 104 gpu, which is not compatiable with 1.4 version of pytorch. This EC2 instance can be accessed using following command.
```bash
ssh -i "deep.pem" ec2-user@ec2-18-204-13-61.compute-1.amazonaws.com
```
