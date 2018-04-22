# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:44:48 2018

@author: allens
"""
import re
def reduce_lengthening( text):
        """It uses the regex to find the pattern and reduce the length"""
        pattern = re.compile(r"(.)\1{2,}")
        return "".join(pattern.sub(r"\1\1", text))
    
tex="[nojira]R251B | D10 | Mod A | UI issues while"
a=reduce_lengthening(tex)    
print(a)


#Initial commit (Auto generated)