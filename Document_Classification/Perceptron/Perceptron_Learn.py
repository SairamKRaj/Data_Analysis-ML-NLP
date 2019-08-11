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
start_time=time.time()

stop_words=['myself', "wasn't", 'a', 'he', 'some', "us", "also", "would", "you've", 'same', 'been', "you'd", "hasn't", 'ma', 'while', 'shouldn', 'me', 'below', "she's", 'wouldn', 'were', 'out', 'own', 'into', 'such', 'at', 'have', 'under', "aren't", 'further', 'they', 'only', 'that', 'your', 'be', 'about', 'our', 'nor', 'its', 'because', 'her', 'any', 'had', 'yourselves', 'those', 'my', 'off', 'themselves', "it's", 'hers', 'up', 'is', 'not', 'both', 'o', 'down', 'am', 'why', 'so', 'haven', 'does', 'just', 'you', 'yours', 'through', "doesn't", 'what', 'too', 'here', 'how', 'all', "shouldn't", 'if', 'of', 'being', 'itself', 'won', 'will', 'after', 'was', 'd', 'himself', 'ours', 'when', 'him', 'theirs', "you'll", 'do', 'or', 'and', "should've", 'over', 'between', 'the', 'other', 'on', 'has', "needn't", 'before', "you're", 'should', 'isn', 'ain', 'to', 'until', 'then', 'against', 'll', 'whom', 'now', 'very', "hadn't", 've', 'there', 'them', "mightn't", 'doesn', "shan't", 'needn', "won't", 'shan', 'for', 'their', 'it', 'i', 'this', 'she', 'yourself', 'by', "isn't", 'above', 'herself', 'where', 'couldn', 'from', 'during', 'can', "don't", 'hasn', 're', 'mustn', 'as', 'than', 'more', 'we', 'again', 'each', 'with', 'are', 'most', "couldn't", 'doing', 'few', "that'll", "weren't", 'aren', 'but', 'did', "mustn't", 'weren', 'his', 'once', 's', 'having', 'don', 'wasn', 'an', 'no', 'these', 'm', 'mightn', 't', 'y', 'who', "wouldn't", 'didn', 'hadn', "haven't", 'which', "didn't", 'in', 'ourselves']


total_docs=0
prior_probability_of_classes=[]
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



total_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
for file in total_files:
    class1, class2, fold, fname = file.split('/')[-4:]
    testing_class_dict[class1 + class2].append(file)
    training_class_dict[class1+class2].append(file)

for file in total_files:
    class1, class2, fold, fname = file.split('/')[-4:]
    if class1=="positive_polarity":
        positive_testing_class_dict[class1].append(file)
    if class1=="negative_polarity":
        negative_testing_class_dict[class1].append(file)
    if (class2=="truthful_from_Web" or class2=="truthful_from_TripAdvisor"):
        truthful_testing_class_dict["truthful"].append(file)
    if (class2=="deceptive_from_MTurk" or class2=="deceptive_from_MTurk"):
        deceptive_testing_class_dict["deceptive"].append(file)

for file in total_files:
    class1, class2, fold, fname = file.split('/')[-4:]
    if class1=="positive_polarity":
        positive_training_class_dict["positive"].append(file)
    if class1=="negative_polarity":
        negative_training_class_dict["negative"].append(file)
    if (class2=="truthful_from_Web" or class2=="truthful_from_TripAdvisor"):
        truthful_training_class_dict["truthful"].append(file)
    if (class2=="deceptive_from_MTurk" or class2=="deceptive_from_MTurk"):
        deceptive_training_class_dict["deceptive"].append(file)


total_vocabulary=[]
total_vocabulary_dict1={}
total_vocabulary_dict2={}
total_vocabulary_dict={}
final_total_vocabulary_dict={}
for class_type, file_list in training_class_dict.items():
    for file in file_list:
        fopen=open(os.path.join(file), "r")
        content=fopen.read()
        fopen.close()
        text_words = content.split()
        text_words = content.split()
        word_table = [word.strip(string.punctuation).lower() for word in text_words]
        #word_table = str.maketrans('', '', string.punctuation)
        #clean_words1 = [word.translate(word_table) for word in text_words]
        clean_words2 = [x for x in word_table if x not in stop_words]
        total_vocabulary.extend(clean_words2)
cnt=Counter(total_vocabulary)
for key, value in cnt.items():
    total_vocabulary_dict1[key] = value
total_vocabulary_dict2=Counter(total_vocabulary_dict1)
total_vocabulary_dict=total_vocabulary_dict2.most_common(2000)
for item in total_vocabulary_dict:
    final_total_vocabulary_dict[item[0]]=item[1]

negative_vocabulary=[]
negative_vocabulary_dict1={}
negative_vocabulary_dict2={}
negative_vocabulary_dict={}
final_negative_vocabulary_dict={}
for class_type, file_list in negative_training_class_dict.items():
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
        negative_vocabulary.extend(clean_words2)
cnt1=Counter(negative_vocabulary)
for key, value in cnt1.items():
    negative_vocabulary_dict1[key] = value
negative_vocabulary_dict2=Counter(negative_vocabulary_dict1)
negative_vocabulary_dict=negative_vocabulary_dict2.most_common(2000)
for item in negative_vocabulary_dict:
    final_negative_vocabulary_dict[item[0]]=item[1]

positive_vocabulary=[]
positive_vocabulary_dict={}
positive_vocabulary_dict1={}
positive_vocabulary_dict2={}
final_positive_vocabulary_dict={}
for class_type, file_list in positive_training_class_dict.items():
    for file in file_list:
        f = open(os.path.join(file), "r")
        content = f.read()
        f.close()
        text_words = content.split()
        text_words = content.split()
        word_table = [word.strip(string.punctuation).lower() for word in text_words]
        #word_table = str.maketrans('', '', string.punctuation)
        #clean_words1 = [word.translate(word_table) for word in text_words]
        clean_words2 = [x for x in word_table if x not in stop_words]
        positive_vocabulary.extend(clean_words2)
cnt4=Counter(positive_vocabulary)
for key, value in cnt4.items():
    positive_vocabulary_dict1[key] = value
positive_vocabulary_dict2=Counter(positive_vocabulary_dict1)
positive_vocabulary_dict=positive_vocabulary_dict2.most_common(2000)
for item in positive_vocabulary_dict:
    final_positive_vocabulary_dict[item[0]]=item[1]

truthful_vocabulary=[]
truthful_vocabulary_dict={}
truthful_vocabulary_dict1={}
truthful_vocabulary_dict2={}
final_truthful_vocabulary_dict={}
for class_type, file_list in truthful_training_class_dict.items():
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
        truthful_vocabulary.extend(clean_words2)
cnt2=Counter(truthful_vocabulary)
for key, value in cnt2.items():
    truthful_vocabulary_dict1[key] = value
truthful_vocabulary_dict2=Counter(truthful_vocabulary_dict1)
truthful_vocabulary_dict=truthful_vocabulary_dict2.most_common(2000)
for item in truthful_vocabulary_dict:
    final_truthful_vocabulary_dict[item[0]]=item[1]


deceptive_vocabulary=[]
deceptive_vocabulary_dict={}
deceptive_vocabulary_dict1={}
deceptive_vocabulary_dict2={}
final_deceptive_vocabulary_dict={}
for class_type, file_list in deceptive_training_class_dict.items():
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
        deceptive_vocabulary.extend(clean_words2)
cnt3=Counter(deceptive_vocabulary)
for key, value in cnt3.items():
    deceptive_vocabulary_dict1[key] = value
deceptive_vocabulary_dict2=Counter(deceptive_vocabulary_dict1)
deceptive_vocabulary_dict=deceptive_vocabulary_dict2.most_common(2000)
for item in deceptive_vocabulary_dict:
    final_deceptive_vocabulary_dict[item[0]]=item[1]

weight_class_positive=0
weight_class_negative=0
weight_class_truthful=0
weight_class_deceptive=0
bias_positive=0
bias_negative=0
bias_truthful=0
bias_deceptive=0

weight_class_averaged_positive=0
weight_class_averaged_negative=0
weight_class_averaged_truthful=0
weight_class_averaged_deceptive=0
bias_averaged_positive=0
bias_averaged_negative=0
bias_averaged_truthful=0
bias_averaged_deceptive=0
cached_weight_class_averaged_positive=0
cached_weight_class_averaged_negative=0
cached_weight_class_averaged_truthful=0
cached_weight_class_averaged_deceptive=0
cached_bias_averaged_positive=0
cached_bias_averaged_negative=0
cached_bias_averaged_truthful=0
cached_bias_averaged_deceptive=0

maxiter=101

y_positive=1
y_negative=-1
count=0

activation_list_positive=[]
activation_list_negative=[]
activation_list_truthful=[]
activation_list_deceptive=[]
positive_weight_list=[]
negative_weight_list=[]
truthful_weight_list=[]
deceptive_weight_list=[]

intermediate_vocabulary=[]
special_total_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
special_training_class_dict = collections.defaultdict(list)
activation_list_positive_dict={}
activation_list_negative_dict={}
activation_list_truthful_dict={}
activation_list_deceptive_dict={}
positive_weight_list_dict={}
negative_weight_list_dict={}
truthful_weight_list_dict={}
deceptive_weight_list_dict={}
averaged_weight_list=[]
averaged_bias_list=[]
averaged_intermediate_vocabulary=[]

counter=1

for i in range(25):
    random.shuffle(special_total_files)
    for file in special_total_files:
        class1, class2, fold, fname = file.split('/')[-4:]
        special_training_class_dict[class1 + class2].append(file)
    for class_type, file_list in special_training_class_dict.items():
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
            intermediate_vocabulary.extend(clean_words2)

            for word1 in intermediate_vocabulary:
                if word1 in final_positive_vocabulary_dict:
                    positive_activation_value = (final_positive_vocabulary_dict[word1]*weight_class_positive)+bias_positive
                    if((y_positive*positive_activation_value)<=0):
                        weight_class_positive=weight_class_positive+(y_positive*final_positive_vocabulary_dict[word1])
                        bias_positive=bias_positive+y_positive
                    activation_list_positive_dict[word1] = positive_activation_value
                    positive_weight_list_dict[word1] = weight_class_positive

                if word1 in final_positive_vocabulary_dict:
                    if(y_positive*(weight_class_averaged_positive*final_positive_vocabulary_dict[word1]+bias_averaged_positive)<=0):
                        weight_class_averaged_positive=weight_class_averaged_positive+(y_positive*final_positive_vocabulary_dict[word1])
                        bias_averaged_positive=bias_averaged_positive+y_positive
                        cached_weight_class_averaged_positive=cached_weight_class_averaged_positive+(y_positive*counter*final_positive_vocabulary_dict[word1])
                        cached_bias_averaged_positive=cached_bias_averaged_positive+(y_positive*counter)

                if word1 in final_truthful_vocabulary_dict:
                    truthful_activation_value = (final_truthful_vocabulary_dict[word1] * weight_class_truthful) + bias_truthful
                    if ((y_positive * truthful_activation_value) <= 0):
                        weight_class_truthful = weight_class_truthful + (y_positive * final_truthful_vocabulary_dict[word1])
                        bias_truthful = bias_truthful + y_positive
                    activation_list_truthful_dict[word1] = truthful_activation_value
                    truthful_weight_list_dict[word1] = weight_class_truthful

                if word1 in final_truthful_vocabulary_dict:
                    if (y_positive * (weight_class_averaged_truthful * final_truthful_vocabulary_dict[word1] + bias_averaged_truthful) <= 0):
                        weight_class_averaged_truthful = weight_class_averaged_truthful + (y_positive * final_truthful_vocabulary_dict[word1])
                        bias_averaged_truthful = bias_averaged_truthful + y_positive
                        cached_weight_class_averaged_truthful = cached_weight_class_averaged_truthful + (y_positive * counter * final_truthful_vocabulary_dict[word1])
                        cached_bias_averaged_truthful = cached_bias_averaged_truthful + (y_positive * counter)

                if word1 in final_negative_vocabulary_dict:
                    negative_activation_value = (final_negative_vocabulary_dict[word1]*weight_class_negative)+bias_negative
                    if((y_negative*negative_activation_value)<=0):
                        weight_class_negative=weight_class_negative+(y_negative*final_negative_vocabulary_dict[word1])
                        bias_negative=bias_negative+y_negative
                    activation_list_negative_dict[word1] = negative_activation_value
                    negative_weight_list_dict[word1] = weight_class_negative

                if word1 in final_negative_vocabulary_dict:
                    if (y_negative * (weight_class_averaged_negative * final_negative_vocabulary_dict[word1] + bias_averaged_negative) <= 0):
                        weight_class_averaged_negative = weight_class_averaged_negative + (y_negative * final_negative_vocabulary_dict[word1])
                        bias_averaged_negative = bias_averaged_negative + y_negative
                        cached_weight_class_averaged_negative = cached_weight_class_averaged_negative + (y_negative * counter * final_negative_vocabulary_dict[word1])
                        cached_bias_averaged_negative = cached_bias_averaged_negative + (y_negative * counter)

                if word1 in final_deceptive_vocabulary_dict:
                    deceptive_activation_value = (final_deceptive_vocabulary_dict[word1] * weight_class_deceptive) + bias_deceptive
                    if ((y_negative * deceptive_activation_value) <= 0):
                        weight_class_deceptive = weight_class_deceptive + (y_negative * final_deceptive_vocabulary_dict[word1])
                        bias_deceptive = bias_deceptive + y_negative
                    activation_list_deceptive_dict[word1] = deceptive_activation_value
                    deceptive_weight_list_dict[word1] = weight_class_deceptive

                if word1 in final_deceptive_vocabulary_dict:
                    if (y_negative * (weight_class_averaged_deceptive * final_deceptive_vocabulary_dict[word1] + bias_averaged_deceptive) <= 0):
                        weight_class_averaged_deceptive = weight_class_averaged_deceptive + (y_negative * final_deceptive_vocabulary_dict[word1])
                        bias_averaged_deceptive = bias_averaged_deceptive + y_negative
                        cached_weight_class_averaged_deceptive = cached_weight_class_averaged_deceptive + (y_negative * counter * final_deceptive_vocabulary_dict[word1])
                        cached_bias_averaged_deceptive = cached_bias_averaged_deceptive + (y_negative * counter)
            intermediate_vocabulary=[]

    counter=counter+1


with open('vanillamodel.txt', 'w') as fopen:
    fopen.write("positive")
    fopen.write("\n")
    fopen.write(str(len(activation_list_positive_dict)))
    fopen.write("\n")
    for key, value in activation_list_positive_dict.items():
        fopen.write(str(key)+":"+str(value))
        fopen.write("\n")
    fopen.write(str(len(positive_weight_list_dict.items())))
    fopen.write("\n")
    for key, value in positive_weight_list_dict.items():
        fopen.write(str(key)+":"+str(value))
        fopen.write("\n")
    fopen.write(str(bias_positive))
    fopen.write("\n")

    fopen.write("truthful")
    fopen.write("\n")
    fopen.write(str(len(activation_list_truthful_dict)))
    fopen.write("\n")
    for key, value in activation_list_truthful_dict.items():
        fopen.write(str(key) + ":" + str(value))
        fopen.write("\n")
    fopen.write(str(len(truthful_weight_list_dict.items())))
    fopen.write("\n")
    for key, value in truthful_weight_list_dict.items():
        fopen.write(str(key) + ":" + str(value))
        fopen.write("\n")
    fopen.write(str(bias_truthful))
    fopen.write("\n")

    fopen.write("negative")
    fopen.write("\n")
    fopen.write(str(len(activation_list_negative_dict)))
    fopen.write("\n")
    for key, value in activation_list_negative_dict.items():
        fopen.write(str(key) + ":" + str(value))
        fopen.write("\n")
    fopen.write(str(len(negative_weight_list_dict.items())))
    fopen.write("\n")
    for key, value in negative_weight_list_dict.items():
        fopen.write(str(key) + ":" + str(value))
        fopen.write("\n")
    fopen.write(str(bias_negative))
    fopen.write("\n")

    fopen.write("deceptive")
    fopen.write("\n")
    fopen.write(str(len(activation_list_deceptive_dict)))
    fopen.write("\n")
    for key, value in activation_list_deceptive_dict.items():
        fopen.write(str(key) + ":" + str(value))
        fopen.write("\n")
    fopen.write(str(len(deceptive_weight_list_dict.items())))
    fopen.write("\n")
    for key, value in deceptive_weight_list_dict.items():
        fopen.write(str(key) + ":" + str(value))
        fopen.write("\n")
    fopen.write(str(bias_deceptive))
fopen.close()

with open('averagedmodel.txt', 'w') as fopen1:
    fopen1.write(str(len(activation_list_positive_dict)))
    fopen1.write("\n")
    for key, value in activation_list_positive_dict.items():
        fopen1.write(str(key) + ":" + str(value))
        fopen1.write("\n")
    fopen1.write(str(len(activation_list_truthful_dict)))
    fopen1.write("\n")
    for key, value in activation_list_truthful_dict.items():
        fopen1.write(str(key) + ":" + str(value))
        fopen1.write("\n")
    fopen1.write(str(len(activation_list_negative_dict)))
    fopen1.write("\n")
    for key, value in activation_list_negative_dict.items():
        fopen1.write(str(key) + ":" + str(value))
        fopen1.write("\n")
    fopen1.write(str(len(activation_list_deceptive_dict)))
    fopen1.write("\n")
    for key, value in activation_list_deceptive_dict.items():
        fopen1.write(str(key) + ":" + str(value))
        fopen1.write("\n")
    fopen1.write(str(weight_class_averaged_positive - ((cached_weight_class_averaged_positive) / counter)))
    fopen1.write("\n")
    fopen1.write(str(weight_class_averaged_truthful - ((cached_weight_class_averaged_truthful) / counter)))
    fopen1.write("\n")
    fopen1.write(str(weight_class_averaged_negative - ((cached_weight_class_averaged_negative) / counter)))
    fopen1.write("\n")
    fopen1.write(str(weight_class_averaged_deceptive - ((cached_weight_class_averaged_deceptive) / counter)))
    fopen1.write("\n")
    fopen1.write(str(bias_averaged_positive-((cached_bias_averaged_positive)/counter)))
    fopen1.write("\n")
    fopen1.write(str(bias_averaged_truthful-((cached_bias_averaged_truthful) / counter)))
    fopen1.write("\n")
    fopen1.write(str(bias_averaged_negative-((cached_bias_averaged_negative) / counter)))
    fopen1.write("\n")
    fopen1.write(str(bias_averaged_deceptive - ((cached_bias_averaged_deceptive) / counter)))

fopen1.close()
print("------%s seconds--------" %(time.time()-start_time))
