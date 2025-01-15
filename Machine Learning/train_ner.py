import spacy
from spacy.training.example import Example

# Training data: a list of tuples containing the text and the named entities
TRAIN_DATA = [
    ("Apple is looking at buying U.K. startup for $1 billion", {'entities': [(0, 5, 'ORG'), (27, 29, 'GPE'), (44, 55, 'MONEY')]}),
    ("Microsoft Corporation is a major player in the tech industry", {'entities': [(0, 18, 'ORG'), (41, 45, 'ORG')]})
]

# Load the blank model
nlp = spacy.blank("en")

# Add the named entity recognizer to the pipeline
ner = nlp.add_pipe("ner", last=True)

# Add the labels for the named entities
ner.add_label("ORG")
ner.add_label("GPE")
ner.add_label("MONEY")

# Begin training
optimizer = nlp.begin_training()
for epoch in range(10):
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5)  # Drop is a regularization parameter

# Save the trained model
nlp.to_disk("ner_model")
