'''
Autor@ Félix Fautsch
Date: 02.06.2025
'''

from PIL import Image
import pytesseract

''' Still Image
source = '/Users/felixfautsch/VS_Python/Projektstudium/OCR_test.png'
print(pytesseract.image_to_string(Image.open(source)))
''' 

import cv2
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import numpy as np
import time
import torch
from torch import nn
from transformers import AutoTokenizer, BertModel
import re
from bs4 import BeautifulSoup
import emoji
import pandas as pd
import time


class BERT_BiGRU(nn.Module):
    def __init__(self, bert_model_name, num_labels, gru_hidden_size=128, dropout_rate=0.3, class_weights=None):
        super(BERT_BiGRU, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.gru = nn.GRU(
            input_size=self.bert.config.hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(gru_hidden_size * 2, num_labels)
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  
        gru_output, _ = self.gru(sequence_output)
        pooled_output = gru_output[:, -1, :]  

        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            return loss, logits
        return logits
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def normalize_split_words(text):
    def fix_word(word):
        if re.fullmatch(r'(?:[a-zA-Z][\W_]{0,2}){2,}[a-zA-Z]', word):
            return re.sub(r'[\W_]+', '', word)
        else:
            return word

    words = text.split()
    words = [fix_word(word) for word in words]
    return ' '.join(words)

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_emojis(text)
    text = normalize_split_words(text)
    return text


model_path = "/Users/felixfautsch/VS_Python/Projektstudium/Bert-BiGRU"
MODEL_NAME = "bert-base-german-cased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
model = BERT_BiGRU(bert_model_name=MODEL_NAME, num_labels=2)  
model.load_state_dict(torch.load(f"{model_path}/model_state_dict.pt", map_location=torch.device("cpu")))
model.to(device)
model.eval() 
print("Modell und Tokenizer erfolgreich geladen!")

font_path = "/Users/felixfautsch/VS_Python/MA_Proposal/Open_Sans/OpenSans-VariableFont_wdth,wght.ttf"
font = ImageFont.truetype(font_path, 20)

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("ReadingBetweenThePixelsBertBiGRU.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Running real-time OCR. Press ESC to exit.")

last_ocr_text = ""
last_predicted_class = None
last_ocr_time = 0
ocr_interval = 4  
position = (20,20) 
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if current_time - last_ocr_time >= ocr_interval:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        last_ocr_text = pytesseract.image_to_string(pil_image)

        if len(last_ocr_text.strip()) > 0:
            last_ocr_text = clean_text(last_ocr_text)
            inputs = tokenizer(
                last_ocr_text, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                logits = model(**inputs)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            last_predicted_class = torch.argmax(probs, dim=-1).item()
            print(f"Text: {last_ocr_text}")
            print(f"Vorhergesagte Klasse: {last_predicted_class}")
            print(f"Hatespeech Wahrscheinlichkeit: {probs[0][1]}")
        else:
            last_predicted_class = None

        last_ocr_time = current_time

    # Always show the last result
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)

    if last_predicted_class == 0:
        text_size = draw.textbbox(position, "NO Hatespeech", font=font)  
        draw.rectangle(text_size, fill="black")
        draw.text(position, f'NO Hatespeech:\n{last_ocr_text.strip()}\nHatespeech probability: {probs[0][1]}', fill="green", font=font)
    elif last_predicted_class == 1:
        text_size = draw.textbbox(position, "HATESPEECH", font=font)  
        draw.rectangle(text_size, fill="black")
        draw.text(position, f'HATESPEECH: {last_ocr_text.strip()}\nHatespeech probability: {probs[0][1]}', fill="red", font=font)
        results.append({"Text": last_ocr_text, "Probability": probs[0][1], "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
})
    else:
        text_size = draw.textbbox(position, "No text detected", font=font)  
        draw.rectangle(text_size, fill="black")
        draw.text(position, 'No text detected', fill="blue", font=font)

    frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    out.write(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
    cv2.imshow("Live OCR Feed", frame_with_text)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

df = pd.DataFrame(results)
#df = df.drop_duplicates(subset=['Text'])
df.to_csv("detected_hatespeech_log.csv", index=False)
print("Logged hate speech instances saved to 'detected_hatespeech_log.csv'.")
cap.release()
out.release()
cv2.destroyAllWindows()