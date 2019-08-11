import sys
import os
import string
import glob
import collections
from collections import Counter
import random
import operator
import numpy as np
import math
import time
import json
start_time=time.time()

word_tag_dict={}
word_tag_dict1={}
word_tag_list=[]
reverse_list=[]
tag_list=[]
text_words=[]
word_list=[]
initial_state_probability_dict={}

def emission_probabilities(emission_probability_list):
    emissions = collections.defaultdict(list)
    for state, emission in emission_probability_list:
        emissions[state].append(emission)
    return {state: probabilities_dict(emission) for state, emission in emissions.items()}

def probabilities_dict(list):
    counts = collections.defaultdict(int)
    for value in list:
        counts[value] =counts[value]+1
    return {key: count * 1.0 / len(list) for key, count in counts.items()}

def markov_dict(tag_list):
    neighbors = collections.defaultdict(list)
    for i in range(len(tag_list) - 1):
        word, neighbor = tag_list[i], tag_list[i+1]
        neighbors[word].append(neighbor)
    return {word: probabilities_dict(neighbors) for word, neighbors in neighbors.items()}

with open(sys.argv[1], "r") as training_file:
    for line in training_file.read().splitlines():
        words_list=list(line.split(" "))
        text_words.extend(words_list)
    cnt1=Counter(text_words).most_common(30000)
    for item in cnt1:
        word_tag=item[0]
        word_tag=word_tag[::-1]
        tag, word= word_tag.split("/", 1)
        tag=tag[::-1]
        word=word[::-1]
        tag_list.append(tag)
        word_list.append(word)
        tagged_word=word+":"+tag
        word_tag_list.append(tagged_word)
    tag_counter=Counter(tag_list)
    for item in tag_counter.items():
        initial_state_probability_dict[item[0]]=item[1]/(sum(tag_counter.values()))

    emissions = emission_probabilities(zip(tag_list, word_list))
    hidden_markov = markov_dict(tag_list)
    initial_state_probability_dict_len=len(initial_state_probability_dict)
    initial_state_probability_dict_len=str(initial_state_probability_dict_len)
    emission_probability_len=len(emissions)
    emission_probability_len=str(emission_probability_len)
    tag_transition_probability_len=len(hidden_markov)
    tag_transition_probability_len=str(tag_transition_probability_len)


with open("hmmmodel.txt", "w") as foutput:
    foutput.write("initial_state_probabilities")
    foutput.write("\n")
    foutput.write(initial_state_probability_dict_len)
    foutput.write("\n")
    foutput.write(json.dumps(initial_state_probability_dict))
    foutput.write("\n")
    foutput.write("Emission Probabilities")
    foutput.write("\n")
    foutput.write(emission_probability_len)
    foutput.write("\n")
    foutput.write(json.dumps(emissions))
    foutput.write("\n")
    foutput.write("Tag transition probability")
    foutput.write("\n")
    foutput.write(tag_transition_probability_len)
    foutput.write("\n")
    foutput.write(json.dumps(hidden_markov))

print("Running Time of program is {}".format(time.time()-start_time))
