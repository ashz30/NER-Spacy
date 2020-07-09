#to be used for spacy custom model creation

from __future__ import unicode_literals, print_function
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import pandas as pd

#model directory and training file locations
MODEL_DIR = "C:\Sharepoint\OneDrive - Blue Prism\impDocuments\hackathon/unstructureddata\model"
TRAIN_CSV = "C:/Users\AEasow\PycharmProjects\spacy\DX Ner\dataset/train.csv"

#convert csv to dataframe
train_df = pd.read_csv(TRAIN_CSV)
#replace blanks for values not defined
train_df.fillna("", inplace = True)

train_data = []

#iterating over columns - all columns except data which has the full text sentence
#this is custom built for dx to make training easier and can be replaced by standard spacy training which gives more flexibility.
#for (columnName, columnData) in train_df.iteritems():
#iterating rows for each column as per columns available
#iterate over rows for each column as label
 #   for index, row in train_df.iterrows():
  #      if(columnName != 'data'):
  #          trainingtext = row[columnName]
   #         # skip over blank rows with "" values
   #         if(trainingtext != ""):
   #             sentence = str(row['data'])
                #print ("sentence : " + sentence + "  Selected text : " + trainingtext + " Label : " + columnName)
    #            start = sentence.find(trainingtext)
      #          end = start + len(trainingtext)
     #           train_data.append((sentence, {"entities": [[start, end, columnName]]}))

train_data = []
for index, row in train_df.iterrows():
    entitiy_var =[]
    sentence = str(row['data'])
    if (sentence != ""):
        for (columnName, columnData) in train_df.iteritems():
            if columnName != 'data':
                trainingtext = row[columnName]
            # skip over blank rows with "" values
                if (trainingtext != ""):
                    #print ("sentence : " + sentence + "  Selected text : " + trainingtext + " Label : " + columnName)
                    start = sentence.find(trainingtext)
                    end = start + len(trainingtext)
                    entitiy_var.append([start, end, columnName])

        train_data.append((sentence, {"entities": entitiy_var}))
print(train_data)


nlp = spacy.blank('en')  # new, empty model. Let’s say it’s for the English language
nlp.vocab.vectors.name = 'bpdxsample'   # give a name to our list of vectors
# add NER pipeline
ner = nlp.create_pipe('ner')  # our pipeline would just do NER
nlp.add_pipe(ner, last=True)  # we add the pipeline to the model

#Data and labels
#To train the model, we’ll need some training data. In the case of product search, these would be queries, where we pre-label entities.

#add labels for each column in our model
for (columnName, columnData) in train_df.iteritems():
    nlp.entity.add_label(columnName)


optimizer = nlp.begin_training()

#for i in range(20):
 #   random.shuffle(DATA)
 #   for text, annotations in DATA:
 #       nlp.update([text], [annotations], sgd=optimizer)

n_iter = 40
# Get names of other pipes to disable them during training to train # only NER and update the weights
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data,
                            size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            # Updating the weights
            nlp.update(texts, annotations, sgd=optimizer,
                       drop=0.35, losses=losses)
        print('Losses', losses)


#  save the model

output_dir = Path(MODEL_DIR)
if not output_dir.exists():
    output_dir.mkdir()
nlp.meta['name'] = "bpdxsamplemodel"  # rename model
nlp.to_disk(output_dir)
print("Saved model to", output_dir)