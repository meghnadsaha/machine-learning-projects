#!/usr/bin/env python
# coding: utf-8

import spacy
import json
from spacy.training.example import Example

# Prepare training data
TRAIN_DATA = [
    ("John Doe works at Google", {"entities": [(0, 8, "PERSON"), (18, 24, "ORG")]}),
    ("Alice is the CEO of Facebook", {"entities": [(0, 5, "PERSON"), (21, 29, "ORG")]}),
    ("I live in New York", {"entities": [(10, 18, "GPE")]})
]

# Create a blank English model
nlp = spacy.blank("en")

# Add the named entity recognizer (NER) to the pipeline
ner = nlp.add_pipe("ner", last=True)

# Add labels to the NER component
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Set up the optimizer
optimizer = nlp.begin_training()

# Training loop
for epoch in range(30):
    losses = {}
    # Shuffle the training data
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)

# Save the trained model
nlp.to_disk("ner_model_from_scratch")

# Load the trained model
nlp = spacy.load("ner_model_from_scratch")

# Test the model on new text passed as argument
import sys
resume_text = sys.argv[1]
doc = nlp(resume_text)

# Extract the recognized entities and print them as a JSON response
entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
print(json.dumps(entities))
