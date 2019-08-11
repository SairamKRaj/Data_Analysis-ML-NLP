import sys
import os
import string
import glob
import collections
import json
import ast
import numpy as np
import math
from collections import defaultdict

vocabulary=[]
prior_class_prob_list1=[]
content_list1=[]
content_list2=[]
prior_class_prob_list2=[]
prior_class_prob_list=[]
content_list=[]
read_lines1=[]
class_types=[]
prior_probabilities=[]
conditional_prob=[]
log_conditional_prob=[]
prior_of_word={}
prior_prob1=0
prior_prob2=0
prior_prob3=0
prior_prob4=0
score={}
scoring=[]
final_values=[]
testing_class_dict=collections.defaultdict(list)
log_conditional=[]
index_value=0
final_class_types=[]

model_file=open('nbmodel.txt', 'r')
read_lines=model_file.readlines()
prior_class_prob_list1=read_lines[0:4]
content_list1=read_lines[5::]
total_files = glob.glob(os.path.join(sys.argv[1], '*\*\*\*.txt'))

for line in prior_class_prob_list1:
    line=line.rstrip()
    prior_class_prob_list2.append(line)

for line in prior_class_prob_list2:
    line=line.split(",")
    prior_class_prob_list.append(line)


for item in prior_class_prob_list:
    class_type, value=item
    class_types.append(class_type)
    prior_probabilities.append(float(value))


for item in prior_probabilities:
    scoring.append(math.log(item))


for class_type in class_types:
    for item in scoring:
        score[class_type]=item


for line in content_list1:
    line=line.rstrip()
    content_list2.append(line)


for line in content_list2:
    line=line.split(",")
    content_list.append(line)


for item in content_list:
    word = item[0]
    prior_prob1=item[1]
    prior_prob2=item[2]
    prior_prob3=item[3]
    prior_prob4=item[4]
    conditional_prob=[prior_prob1, prior_prob2, prior_prob3, prior_prob4]
    prior_of_word[word]=conditional_prob

for file in total_files:
    class1, class2, fold, fname = file.split('\\')[-4:]
    testing_class_dict[class1+class2].append(file)


for class_type, file_list in testing_class_dict.items():
    f = open('nboutput.txt', 'w')
    for file in file_list:
        fopen=open(os.path.join(file), "r")
        content=fopen.read()
        text_words = content.split()
        text_words = [word.lower() for word in text_words]
        word_table = str.maketrans('', '', string.punctuation)
        clean_words = [word.translate(word_table) for word in text_words]
        vocabulary.extend(clean_words)


        for word in vocabulary:
            if word in prior_of_word:
                value=prior_of_word.get(word)
                max_value=max(value)
                max_index = value.index(max_value)
        final_class_types=class_types[max_index]


        if final_class_types == "negative_polaritydeceptive_from_MTurk":
            f.write("deceptive"+" "+"negative"+" "+fopen.name+"\n")
        elif final_class_types == "negative_polaritytruthful_from_Web":
            f.write("truthful"+" "+"negative"+" "+fopen.name+"\n")
        elif final_class_types == "positive_polaritydeceptive_from_MTurk":
            f.write("deceptive"+" "+"positive"+" "+fopen.name+"\n")
        elif final_class_types == "positive_polaritytruthful_from_TripAdvisor":
            f.write("truthful"+" "+"positive"+" "+fopen.name+"\n")

    text_words.clear()
    clean_words.clear()
    vocabulary.clear()
