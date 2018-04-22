# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:19:15 2018
 
"""


import random
from pathlib import Path
import spacy

class Train_NER:
    
    LABEL = ''
    
    LABEL_LIST = []
    
    NUM_OF_ITER = 10
    
    def __init__(self):
        return;
        
    def get_sentences(self,file):
        file_reader = open(file, 'r')
        content = file_reader.readlines()
        paragraph = [x.strip().lower().split(".") for x in content]
        sentences = []
        for line in paragraph:
            for sentence in line:
                sentences.append(sentence)
        return sentences
    
    def get_train_data(self, file):
        nlp = spacy.load('en_core_web_sm')
        TRAIN_DATA = []
        nouns_dict = dict()
        sentences = self.get_sentences(file)
        for line in sentences:
            count = 0
            if("**" in line[:2]):
                self.LABEL = line.upper()
                self.LABEL = self.LABEL[2: ]
                self.LABEL_LIST.append(self.LABEL)
            elif(len(line)!=0):
                i =0
                doc = nlp(line)
                for token in doc:
                    if (token.pos_ is "NOUN" or token.pos_ is "PROPN"):
                        start_index = line.find(token.text, i)
                        end_index = start_index + len(token.text) 
                        if(count == 0):
                            nouns_dict['entities'] = [(start_index ,end_index , self.LABEL)] 
                            count = count + 1
                        else :
                            nouns_dict['entities'].append((start_index ,end_index , self.LABEL))
                        entity_dict = nouns_dict.items()
                        train_data_tuple = line,dict(entity_dict)
                    i = i + len(token.text)
                TRAIN_DATA.append(train_data_tuple)  
                
                nouns_dict.clear()
        print (TRAIN_DATA)
        return TRAIN_DATA
    
    
    def train(self, file, output_dir, n_iter=NUM_OF_ITER, model='samp/'):
        """Set up the pipeline and entity recognizer, and train the new entity."""
        if model is not None:
            nlp = spacy.load(model)  # load existing spaCy model
            #print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank('en')  # create blank Language class
            #print("Created blank 'en' model")
        TRAIN_DATA = self.get_train_data(file)
        # Add entity recognizer to model if it's not in the pipeline
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        else:
            ner = nlp.get_pipe('ner')
        
        # Adding labels to NER.
        for label in self.LABEL_LIST:
            ner.add_label(label) 
            
        # Get names of other pipes to disable them during training.
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in TRAIN_DATA:
                    nlp.update([text], [annotations], sgd=optimizer, drop=0.35,losses=losses)
                print(losses)
                
        # Save model to output directory.
        if output_dir is not None:
            direc = Path(output_dir)
            if not direc.exists():
                direc.mkdir()
            nlp.to_disk(direc)
            #print("Saved model to", output_dir)
            
        # Writing Labels to a file
        file_name = 'labels.txt'
        file_writer = open(output_dir+file_name, "w")
        for label in self.LABEL_LIST:
            file_writer.write(label+'\n')
        file_writer.close()
        #print(output_dir)
            
NER = Train_NER()
#print('heyy')
NER.train('RegulatoryDoc.txt','combined/')
            
            
            
            
            
            
            
