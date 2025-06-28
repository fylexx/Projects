# This directory contains the results from my research on automatic hatespeech detection with transformers.
The notebook contains the the necessary steps for the BERT-SAGEConv architecture. Besides the geometric implementation, other BERT extensions are also part of the notebook (CNN, biLSTM, RNN, FC, GATConv). The focus however lies on the co-occurence graph creation and integrating those features with the BERT embeddings. The inspiration for this project comes from: *Yang, Y., & Cui, X. (2021). Bert-enhanced text graph neural network for classification. Entropy, 23(11), 1536* and was part of the *Projektstudium* for my *M.Sc. Information Systems and Management* degree.

![BERT SAGEConv Architecture](https://github.com/fylexx/Projects/blob/main/Hatespeech/BERT-SAGEConv_Architecture.png)

*a) Preprocessing data and creating vocabular for graps; b) BERT Feature Extraction; c) Graph Creation and Feature Extraction with SAGEConv; d) Feature Concatenation and Aggregation into fully connected network with softmax for Hatespeech classification*
