import torch
import streamlit as st

import random
import re
import os
import string
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

from simplet5 import SimpleT5

torch.cuda.empty_cache()

st.title("Abstractive Text summarization")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_GPU = None
if str(DEVICE) == "cuda":
    USE_GPU=True
else:
    USE_GPU = False

class T5Model:
    def __init__(self, model_type, model_name):
        self.model = SimpleT5()
        self.model.from_pretrained(model_type=model_type,
                                   model_name=model_name)

    def load_model(self, model_type, model_path, use_gpu: bool):
        try:
            self.model.load_model(
                model_type=model_type,
                model_dir=model_path,
                use_gpu=use_gpu
            )

        except BaseException as ex:
            print("error occurred while loading model ", str(ex))

text_to_summarize = st.text_area("Enter Text","")

best_weight = "C:/Users/Alana/Desktop/miniproject/outputs"

t5_model = T5Model(model_name="t5-base",model_type="t5")
t5_model.load_model(model_type="t5",
                   use_gpu=USE_GPU,
                   model_path=best_weight
                   )


if st.button('Summarize'):
    c_result = t5_model.model.predict(text_to_summarize)

    c_result=str(c_result)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', c_result)
    if len(sentences) > 0 and not sentences[-1].endswith('.'):
        # Remove the last sentence from the paragraph
        c_result = ' '.join(sentences[:-1])
        sentences =c_result.split('. ')
    unique_sentences = []
    for sentence in sentences:
        # Strip leading and trailing whitespaces from each sentence
            sentence = sentence.strip()
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
     # Join the unique sentences with a period and space
    c_result = '. '.join(unique_sentences)
    st.success(c_result)
if st.button('Keywords'):
    words = word_tokenize(text_to_summarize)
    
    # Filter out stopwords (common words that don't carry much meaning)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    
    # Perform part-of-speech tagging
    tagged_words = nltk.pos_tag(filtered_words)
    num_keywords = 3

    # Extract keywords based on specific part-of-speech patterns
    keywords = [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('JJ')]
    keyword_freq = Counter(keywords)
    top_keywords = keyword_freq.most_common(num_keywords)
    key=[]
    key=[word for word, _ in top_keywords]
    st.success(key)

