import sys
import os
import string
import glob
import collections
from collections import Counter
import random
import operator
import math
import time
start_time=time.time()

stop_words=['myself', "wasn't", 'a', 'he', 'some', "us", "also", "would", "you've", 'same', 'been', "you'd", "hasn't", 'ma', 'while', 'shouldn', 'me', 'below', "she's", 'wouldn', 'were', 'out', 'own', 'into', 'such', 'at', 'have', 'under', "aren't", 'further', 'they', 'only', 'that', 'your', 'be', 'about', 'our', 'nor', 'its', 'because', 'her', 'any', 'had', 'yourselves', 'those', 'my', 'off', 'themselves', "it's", 'hers', 'up', 'is', 'not', 'both', 'o', 'down', 'am', 'why', 'so', 'haven', 'does', 'just', 'you', 'yours', 'through', "doesn't", 'what', 'too', 'here', 'how', 'all', "shouldn't", 'if', 'of', 'being', 'itself', 'won', 'will', 'after', 'was', 'd', 'himself', 'ours', 'when', 'him', 'theirs', "you'll", 'do', 'or', 'and', "should've", 'over', 'between', 'the', 'other', 'on', 'has', "needn't", 'before', "you're", 'should', 'isn', 'ain', 'to', 'until', 'then', 'against', 'll', 'whom', 'now', 'very', "hadn't", 've', 'there', 'them', "mightn't", 'doesn', "shan't", 'needn', "won't", 'shan', 'for', 'their', 'it', 'i', 'this', 'she', 'yourself', 'by', "isn't", 'above', 'herself', 'where', 'couldn', 'from', 'during', 'can', "don't", 'hasn', 're', 'mustn', 'as', 'than', 'more', 'we', 'again', 'each', 'with', 'are', 'most', "couldn't", 'doing', 'few', "that'll", "weren't", 'aren', 'but', 'did', "mustn't", 'weren', 'his', 'once', 's', 'having', 'don', 'wasn', 'an', 'no', 'these', 'm', 'mightn', 't', 'y', 'who', "wouldn't", 'didn', 'hadn', "haven't", 'which', "didn't", 'in', 'ourselves']


clean_words2=[]
token_count=[]
final_token_count={}
final_token_count1={}
final_token_count2={}
final_token_count3={}
final_token_count4={}
final_word_count1=0
final_word_count2=0
final_word_count3=0
final_word_count4=0
final_word_count=0
total_term_count=0
total_terms=0
final_clean_words=[]
final_clean_words_set=[]
full_final_clean_words_set=[]
prior_probability_of_class = collections.defaultdict(list)
positive_deceptive_training_class_dict=collections.defaultdict(list)
positive_deceptive_testing_class_dict=collections.defaultdict(list)
positive_truthful_training_class_dict=collections.defaultdict(list)
positive_truthful_testing_class_dict=collections.defaultdict(list)
negative_deceptive_training_class_dict=collections.defaultdict(list)
negative_deceptive_testing_class_dict=collections.defaultdict(list)
negative_truthful_training_class_dict=collections.defaultdict(list)
negative_truthful_testing_class_dict=collections.defaultdict(list)
training_class_dict=collections.defaultdict(list)
positive_training_class_dict=collections.defaultdict(list)
negative_training_class_dict=collections.defaultdict(list)
truthful_training_class_dict=collections.defaultdict(list)
deceptive_training_class_dict=collections.defaultdict(list)
testing_class_dict=collections.defaultdict(list)
positive_testing_class_dict=collections.defaultdict(list)
negative_testing_class_dict=collections.defaultdict(list)
truthful_testing_class_dict=collections.defaultdict(list)
deceptive_testing_class_dict=collections.defaultdict(list)
training_positive_class_activation_dict={}
training_negative_class_activation_dict={}
training_deceptive_class_activation_dict={}
training_truthful_class_activation_dict={}
training_positive_class_weight_dict={}
training_negative_class_weight_dict={}
training_deceptive_class_weight_dict={}
training_truthful_class_weight_dict={}
content1=[]
content2=[]
content3=[]
content4=[]
positive_bias=0
truthful_bias=0
negative_bias=0
deceptive_bias=0
positive_averaged_bias=0
truthful_averaged_bias=0
negative_averaged_bias=0
deceptive_averaged_bias=0

positive_activation_value=0
negative_activation_value=0
truthful_activation_value=0
deceptive_activation_value=0
positive_averaged_activation_value=0
negative_averaged_activation_value=0
truthful_averaged_activation_value=0
deceptive_averaged_activation_value=0

positive_activation_value_dict={}
negative_activation_value_dict={}
truthful_activation_value_dict={}
deceptive_activation_value_dict={}
averaged_positive_activation_value_dict={}
averaged_negative_activation_value_dict={}
averaged_truthful_activation_value_dict={}
averaged_deceptive_activation_value_dict={}

final_activation_value_list1=[]
final_activation_value_list2=[]
testing_intermediate_vocabulary=[]
testing_vocabulary=[]
testing_vocabulary_dict1={}
testing_vocabulary_dict2={}
testing_vocabulary_dict={}



total_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))
for file in total_files:
    class1, class2, fold, fname = file.split('/')[-4:]
    if fold == "fold1":
        testing_class_dict[class1 + class2].append(file)


for class_type, file_list in testing_class_dict.items():
    for file in file_list:
        f = open(os.path.join(file), "r")
        content = f.read()
        f.close()
        text_words = content.split()
        word_table = [word.strip(string.punctuation).lower() for word in text_words]
        #word_table = str.maketrans('', '', string.punctuation)
        #clean_words1 = [word.translate(word_table) for word in text_words]
        clean_words2 = [x for x in word_table if x not in stop_words]
        testing_vocabulary.extend(clean_words2)
cnt=Counter(testing_vocabulary)
for key, value in cnt.items():
    testing_vocabulary_dict1[key] = value
testing_vocabulary_dict2=Counter(testing_vocabulary_dict1)
#print(testing_vocabulary_dict2)

averaged_weight_list=[]
averaged_bias_list=[]
positive_averaged_weight=0
truthful_averaged_weight=0
negative_averaged_weight=0
deceptive_averaged_weight=0

model_file=sys.argv[1]
if (model_file == "vanillamodel.txt"):
    fopen = open(model_file,"r")
    positive_class_type=fopen.readline()
    positive_class_activation_counts=fopen.readline()
    for i in range(int(positive_class_activation_counts)):
        content1=tuple(fopen.readline().split(":"))
        training_positive_class_activation_dict[content1[0]]=int(content1[1])
    positive_class_weight_counts = fopen.readline()
    for i in range(int(positive_class_weight_counts)):
        content2=tuple(fopen.readline().split(":"))
        training_positive_class_weight_dict[content2[0]] = int(content2[1])
    positive_bias=fopen.readline()
    truthful_class_type = fopen.readline()
    truthful_class_activation_counts = fopen.readline()
    for i in range(int(truthful_class_activation_counts)):
        content3 = tuple(fopen.readline().split(":"))
        training_truthful_class_activation_dict[content3[0]] = int(content3[1])
    truthful_class_weight_counts = fopen.readline()
    for i in range(int(truthful_class_weight_counts)):
        content4 = tuple(fopen.readline().split(":"))
        training_truthful_class_weight_dict[content4[0]] = int(content4[1])
    truthful_bias = fopen.readline()
    negative_class_type = fopen.readline()
    negative_class_activation_counts = fopen.readline()
    for i in range(int(negative_class_activation_counts)):
        content5 = tuple(fopen.readline().split(":"))
        training_negative_class_activation_dict[content5[0]] = int(content5[1])
    negative_class_weight_counts = fopen.readline()
    for i in range(int(negative_class_weight_counts)):
        content6 = tuple(fopen.readline().split(":"))
        training_negative_class_weight_dict[content6[0]] = int(content6[1])
    negative_bias = fopen.readline()

    deceptive_class_type = fopen.readline()
    deceptive_class_activation_counts = fopen.readline()
    for i in range(int(deceptive_class_activation_counts)):
        content7 = tuple(fopen.readline().split(":"))
        training_deceptive_class_activation_dict[content7[0]] = int(content7[1])
    deceptive_class_weight_counts = fopen.readline()
    for i in range(int(deceptive_class_weight_counts)):
        content8 = tuple(fopen.readline().split(":"))
        training_deceptive_class_weight_dict[content8[0]] = int(content8[1])
    deceptive_bias = fopen.readline()

    with open("percepoutput.txt", "w") as foutput:
        for class_type, file_list in testing_class_dict.items():
            for file in file_list:
                f = open(os.path.join(file), "r")
                content = f.read()
                f.close()
                text_words = content.split()
                text_words = [word.lower() for word in text_words]
                text_words = content.split()
                word_table = [word.strip(string.punctuation).lower() for word in text_words]
                #word_table = str.maketrans('', '', string.punctuation)
                #clean_words1 = [word.translate(word_table) for word in text_words]
                clean_words2 = [x for x in word_table if x not in stop_words]
                testing_intermediate_vocabulary.extend(set(clean_words2))

                for word1 in testing_intermediate_vocabulary:
                    if word1 in training_positive_class_activation_dict:
                        positive_activation_value = int(training_positive_class_activation_dict[word1]) * int(testing_vocabulary_dict2[word1]) + int(positive_bias)

                for word1 in testing_intermediate_vocabulary:
                    if word1 in training_negative_class_weight_dict:
                        negative_activation_value = (int(training_negative_class_activation_dict[word1]) * int(testing_vocabulary_dict2[word1])) + int(negative_bias)


                for word1 in testing_intermediate_vocabulary:
                    if word1 in training_truthful_class_weight_dict:
                        truthful_activation_value = (int(training_truthful_class_activation_dict[word1]) * int(testing_vocabulary_dict2[word1])) + int(truthful_bias)


                for word1 in testing_intermediate_vocabulary:
                    if word1 in training_deceptive_class_weight_dict:
                        deceptive_activation_value = (int(training_deceptive_class_activation_dict[word1]) * int(testing_vocabulary_dict2[word1])) + int(deceptive_bias)

                
                testing_intermediate_vocabulary=[]
                
                negative_activation_value = -negative_activation_value
                deceptive_activation_value = -deceptive_activation_value

                if ((positive_activation_value>negative_activation_value) and (truthful_activation_value>deceptive_activation_value)):
                    foutput.write("truthful" + " " + "positive" + " " + file + "\n")
                if ((positive_activation_value < negative_activation_value) and (
                        truthful_activation_value > deceptive_activation_value)):
                    foutput.write("truthful" + " " + "negative" + " " + file + "\n")
                if ((positive_activation_value > negative_activation_value) and (
                        truthful_activation_value < deceptive_activation_value)):
                    foutput.write("deceptive" + " " + "positive" + " " + file + "\n")
                if ((positive_activation_value < negative_activation_value) and (
                        truthful_activation_value < deceptive_activation_value)):
                    foutput.write("deceptive" + " " + "negative" + " " + file + "\n")
                

elif (model_file == "averagedmodel.txt"):
    fopen = open(model_file, "r")
    positive_class_activation_counts = fopen.readline()
    for i in range(int(positive_class_activation_counts)):
        content1 = tuple(fopen.readline().split(":"))
        training_positive_class_activation_dict[content1[0]] = int(content1[1])

    truthful_class_activation_counts = fopen.readline()
    for i in range(int(truthful_class_activation_counts)):
        content1 = tuple(fopen.readline().split(":"))
        training_truthful_class_activation_dict[content1[0]] = int(content1[1])

    negative_class_activation_counts = fopen.readline()
    for i in range(int(negative_class_activation_counts)):
        content1 = tuple(fopen.readline().split(":"))
        training_negative_class_activation_dict[content1[0]] = int(content1[1])

    deceptive_class_activation_counts = fopen.readline()
    for i in range(int(deceptive_class_activation_counts)):
        content1 = tuple(fopen.readline().split(":"))
        training_deceptive_class_activation_dict[content1[0]] = int(content1[1])
    positive_averaged_weight=float(fopen.readline())
    truthful_averaged_weight=float(fopen.readline())
    negative_averaged_weight=float(fopen.readline())
    deceptive_averaged_weight=float(fopen.readline())
    positive_averaged_bias=float(fopen.readline())
    truthful_averaged_bias=float(fopen.readline())
    negative_averaged_bias=float(fopen.readline())
    deceptive_averaged_bias=float(fopen.readline())

    with open("percepoutput.txt", "w") as foutput:
        for class_type, file_list in testing_class_dict.items():
            for file in file_list:
                f = open(os.path.join(file), "r")
                content = f.read()
                f.close()
                text_words = content.split()
                text_words = [word.lower() for word in text_words]
                word_table = [word.strip(string.punctuation).lower() for word in text_words]
                # word_table = str.maketrans('', '', string.punctuation)
                # clean_words1 = [word.translate(word_table) for word in text_words]
                clean_words2 = [x for x in word_table if x not in stop_words]
                testing_intermediate_vocabulary.extend(set(clean_words2))

                for word1 in testing_intermediate_vocabulary:
                    if word1 in training_positive_class_activation_dict:
                        positive_averaged_activation_value = (positive_averaged_weight*testing_vocabulary_dict2[word1])+positive_averaged_bias

                    if word1 in training_negative_class_activation_dict:
                        negative_averaged_activation_value = (negative_averaged_weight*testing_vocabulary_dict2[word1])+negative_averaged_bias

                    if word1 in training_truthful_class_activation_dict:
                        truthful_averaged_activation_value = (truthful_averaged_weight*testing_vocabulary_dict2[word1])+truthful_averaged_bias

                    if word1 in training_deceptive_class_activation_dict:
                        deceptive_averaged_activation_value = (deceptive_averaged_weight*testing_vocabulary_dict2[word1])+deceptive_averaged_bias
                    
                testing_intermediate_vocabulary=[]

                negative_averaged_activation_value = -negative_averaged_activation_value
                deceptive_averaged_activation_value = -deceptive_averaged_activation_value

                if ((positive_averaged_activation_value > negative_averaged_activation_value) and (truthful_averaged_activation_value > deceptive_averaged_activation_value)):
                    foutput.write("truthful" + " " + "positive" + " " + file + "\n")
                if ((positive_averaged_activation_value < negative_averaged_activation_value) and (truthful_averaged_activation_value > deceptive_averaged_activation_value)):
                    foutput.write("truthful" + " " + "negative" + " " + file + "\n")
                if ((positive_averaged_activation_value > negative_averaged_activation_value) and (truthful_averaged_activation_value < deceptive_averaged_activation_value)):
                    foutput.write("deceptive" + " " + "positive" + " " + file + "\n")
                if ((positive_averaged_activation_value < negative_averaged_activation_value) and (truthful_averaged_activation_value < deceptive_averaged_activation_value)):
                    foutput.write("deceptive" + " " + "negative" + " " + file + "\n")


print("------%s seconds--------" %(time.time()-start_time))
