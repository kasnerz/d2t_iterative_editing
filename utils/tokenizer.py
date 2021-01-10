#!/usr/bin/env python

from sacremoses import MosesDetokenizer
from nltk import word_tokenize, sent_tokenize
import re

class Tokenizer:
    def __init__(self):
        self.detokenizer = MosesDetokenizer(lang='en')

    def tokenize(self, s):
        tokens = []

        for sentence in sent_tokenize(s):
            # remove underscores
            sentence = re.sub(r'_', r' ', sentence)

            # split basic camel case, lowercase first letters
            sentence = re.sub(r"([a-z])([A-Z])",
                lambda m: rf"{m.group(1)} {m.group(2).lower()}", sentence)

            # NLTK word tokenize
            tokens += word_tokenize(sentence)

        res = " ".join(tokens)
        return res


    def detokenize(self, s):
        tokens = s.split()

        return self.detokenizer.detokenize(tokens)