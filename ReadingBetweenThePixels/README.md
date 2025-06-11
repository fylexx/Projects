# _Reading Between the Pixels_: OCR and BERT-BiGRU for Live Hate-Speech Detection
### Combining OCR (Optical Character Recognition) with a finetuned and augmented BERT model for real time hatespeech detection
### Combination of Computer Vision and NLP
---
Using a **hybrid BERT Architecture**, combining BERT with one bidirectional GRU layer (BiGRU). The model has been trained on the balanced superset ver1.5 using cross validation.<br>
For OCR I have used **Pytesseract** for its responsivness and efficient handling of multiple languages.<br>
The process of the model is presented in the graph down below. Overall we conduct an OCR scan every 4 seconds (for CPU efficiency). If text was detected, the finetuned BERT-BiGRU model classifies it into either hatespeech or no hatespeech.

---
![Reading Between the Pixels_: OCR and BERT-BiGRU for Live Hate-Speech Detection Architecture](ReadingBetweenThePixels/ReadingBetweenThePixelsArchitecture.png)
