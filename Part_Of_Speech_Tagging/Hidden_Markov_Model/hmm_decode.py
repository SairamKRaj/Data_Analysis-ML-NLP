import collections
from collections import Counter
import random
import operator
import numpy as np
import math
import itertools
import sys
import json
import time
start_time=time.time()
initial_state_probability_dict={}
emission_probability_dict={}
value1=[]
words_list=[]
tag_transition_probability_dict={}
INIT_STATE = 'init'
FINAL_STATE = 'final'



#Read hmmmodel.txt file
finput=open("hmmmodel.txt", "r")
initial_state_probabilities=finput.readline()
initial_state_probability_dict_len=int(finput.readline())
initial_state_probability_dict=dict(json.loads(finput.readline()))
tag_list=list(initial_state_probability_dict.keys())
N=len(tag_list)
enumerated_tag_list=[]
for k,v in enumerate(tag_list,1):
    enumerated_tag_list.append((k,v))
emission_probabilities=finput.readline()
emission_probabilities_len=int(finput.readline())
emission_probability_dict=dict(json.loads(finput.readline()))
tag_transition_probability=finput.readline()
tag_transition_probability_len=int(finput.readline())
tag_transition_probability_dict=dict(json.loads(finput.readline()))
finput.close()
count=0
prev_viterbi_dict={}
resultant_list=[]

foutput=open("hmmoutput.txt", "w")

with open(sys.argv[1], "r") as testing_file:
    for line in testing_file.read().splitlines():
        backpointer = []
        content=list(line.split())
        for word in content:
            words_list.append(word)
            T=len(words_list)
            viterbi_dict = {}
        for state in tag_list:
            viterbi_dict[(state, 0)]=abs(math.log(initial_state_probability_dict[state] or 1)+math.log(emission_probability_dict[state].get(words_list[0]) or 1))
        max_value = (max(viterbi_dict.items(), key=operator.itemgetter(1))[1])
        max_state = (max(viterbi_dict.items(), key=operator.itemgetter(1))[0])
        state1=max_state[0]
        backpointer.append(state1)
        prev_viterbi_dict=viterbi_dict
        viterbi_dict={}
        for time_step in range(1,T):
            for state in tag_list:
                viterbi_dict[(state, time_step)]=abs(math.log(abs(prev_viterbi_dict[(state1, time_step-1)]) or 1)+math.log(float(tag_transition_probability_dict[state1].get(state) or 1.2))+math.log(float(emission_probability_dict[state].get(words_list[time_step]) or 1)))
            max_value = max(viterbi_dict.items(), key=operator.itemgetter(1))[1]
            max_state = max(viterbi_dict.items(), key=operator.itemgetter(1))[0]
            state1 = max_state[0]
            backpointer.append(state1)
            prev_viterbi_dict={}
            prev_viterbi_dict = viterbi_dict
            viterbi_dict = {}


        i = 0
        res_str = ""
        for word in words_list:
            res_str = res_str+(str(word)+"/"+str(backpointer[i])+" ")
            i = i + 1
        foutput.write(res_str)
        foutput.write("\n")
        
        words_list=[]

print("Running Time of program is {}".format(time.time()-start_time))
