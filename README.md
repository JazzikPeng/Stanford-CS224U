# Summary
Sperate semantic meaning from language representation models (e.g. BERT). This is a more fundamental topic compares to the first one. Here is the hypothesis. If we use BERT to encode a sentence, the vector will contain both semantic information and syntactic information. Here is one example. 

    Target: I love to play video games.

    Candidate 1: Gaming is my favorite hobby.

    Candidate 2: I love to eat apples.

Which candidate is more similar to Target? Semantically, candidate 1 clearly wins, but syntactically, candidate 2 might be closer. This can be a very difficult question for language representation models like BERT. We want to train a model to find a sub-vector that contains only semantic information. There are various applications for this, such as redundancy detection. 