# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:40:58 2018

 
"""


import spacy

class Test_NER:
    
    def get_labels(self, model):
        labels = []
        file_name = 'labels.txt'
        file_reader = open(model+file_name, "r")
        for label in file_reader:
            labels.append(''.join(label).strip())
        file_reader.close()
        return labels
            
        
    def test(self, model, comment):
        entities = []
        texts = []
        nlp = spacy.load(model)
        doc = nlp(comment)
        for ent in doc.ents:
            #print(ent.label_, ent.text )
            texts.append(ent.text)
            entities.append(ent.label_.strip()) 
        labels = self.get_labels(model)
        score = []
        for ent in entities:
                if ent in labels:
                    score.append(ent)
        return score,texts
    
N = Test_NER()
print(N.test('combined/','Exceptions should at least have a message'))
        
        
        
