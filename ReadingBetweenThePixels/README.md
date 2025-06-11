# _Reading Between the Pixels_: OCR and BERT-BiGRU for Live Hate-Speech Detection
### Combining OCR (Optical Character Recognition) with a finetuned and augmented BERT model for real time hatespeech detection
### Combination of Computer Vision and NLP
---
Using a **hybrid BERT Architecture**, combining BERT with one bidirectional GRU layer (BiGRU). The model has been trained on a self created superset, consisting of the available german hatespeech datasets. The dataset as well as the notebook for creating the dataset can also be found here. The notebook for training the BERT-BiGRU architecture on the balanced superset is also in this repository.<br>
For OCR I have used **Pytesseract** for its responsivness and efficient handling of multiple languages.<br>
The process of the model is presented in the graph down below. Overall we conduct an OCR scan every 4 seconds (for CPU efficiency). If text was detected, the finetuned BERT-BiGRU model classifies it into either hatespeech or no hatespeech.

---
![Reading Between the Pixels_: OCR and BERT-BiGRU for Live Hate-Speech Detection Architecture](https://github.com/fylexx/Projects/blob/main/ReadingBetweenThePixels/ReadingBetweenThePixelsArchitecture.png)
