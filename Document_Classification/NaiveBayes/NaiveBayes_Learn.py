import sys
import os
import string
import glob
import collections
from collections import Counter

training_class_dict=collections.defaultdict(list)
testing_class_dict=collections.defaultdict(list)
vocabulary=[]
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

stop_words=['myself', "wasn't", 'a', 'he', 'some', "us", "also", "would", "you've", 'same', 'been', "you'd", "hasn't", 'ma', 'while', 'shouldn', 'me', 'below', "she's", 'wouldn', 'were', 'out', 'own', 'into', 'such', 'at', 'have', 'under', "aren't", 'further', 'they', 'only', 'that', 'your', 'be', 'about', 'our', 'nor', 'its', 'because', 'her', 'any', 'had', 'yourselves', 'those', 'my', 'off', 'themselves', "it's", 'hers', 'up', 'is', 'not', 'both', 'o', 'down', 'am', 'why', 'so', 'haven', 'does', 'just', 'you', 'yours', 'through', "doesn't", 'what', 'too', 'here', 'how', 'all', "shouldn't", 'if', 'of', 'being', 'itself', 'won', 'will', 'after', 'was', 'd', 'himself', 'ours', 'when', 'him', 'theirs', "you'll", 'do', 'or', 'and', "should've", 'over', 'between', 'the', 'other', 'on', 'has', "needn't", 'before', "you're", 'should', 'isn', 'ain', 'to', 'until', 'then', 'against', 'll', 'whom', 'now', 'very', "hadn't", 've', 'there', 'them', "mightn't", 'doesn', "shan't", 'needn', "won't", 'shan', 'for', 'their', 'it', 'i', 'this', 'she', 'yourself', 'by', "isn't", 'above', 'herself', 'where', 'couldn', 'from', 'during', 'can', "don't", 'hasn', 're', 'mustn', 'as', 'than', 'more', 'we', 'again', 'each', 'with', 'are', 'most', "couldn't", 'doing', 'few', "that'll", "weren't", 'aren', 'but', 'did', "mustn't", 'weren', 'his', 'once', 's', 'having', 'don', 'wasn', 'an', 'no', 'these', 'm', 'mightn', 't', 'y', 'who', "wouldn't", 'didn', 'hadn', "haven't", 'which', "didn't", 'in', 'ourselves']

positive_deceptive_training_class_dict=collections.defaultdict(list)
positive_deceptive_testing_class_dict=collections.defaultdict(list)
positive_truthful_training_class_dict=collections.defaultdict(list)
positive_truthful_testing_class_dict=collections.defaultdict(list)
negative_deceptive_training_class_dict=collections.defaultdict(list)
negative_deceptive_testing_class_dict=collections.defaultdict(list)
negative_truthful_training_class_dict=collections.defaultdict(list)
negative_truthful_testing_class_dict=collections.defaultdict(list)

total_files = glob.glob(os.path.join(sys.argv[1], '*\*\*\*.txt'))
negative_deceptive_classifier_files = glob.glob(os.path.join(sys.argv[1], 'negative_polarity\deceptive_from_MTurk\*\*.txt'))
negative_truthful_classifier_files = glob.glob(os.path.join(sys.argv[1], 'negative_polarity\truthful_from_Web\*\*.txt'))
positive_deceptive_classifier_files = glob.glob(os.path.join(sys.argv[1], 'positive_polarity\deceptive_from_MTurk\*\*.txt'))
positive_truthful_classifier_files = glob.glob(os.path.join(sys.argv[1], 'positive_polarity\truthful_from_TripAdvisor\*\*.txt'))


for file in total_files:
    class1, class2, fold, fname = file.split('\\')[-4:]
    testing_class_dict[class1+class2].append(file)
    training_class_dict[class1+class2].append(file)

for file in negative_deceptive_classifier_files:
    class1, class2, fold, fname = file.split('\\')[-4:]
    negative_deceptive_testing_class_dict[class1+class2].append(file)
    negative_deceptive_training_class_dict[class1+class2].append(file)

for file in negative_truthful_classifier_files:
    class1, class2, fold, fname = file.split('\\')[-4:]
    negative_truthful_testing_class_dict[class1+class2].append(file)
    negative_truthful_training_class_dict[class1+class2].append(file)

for file in positive_deceptive_classifier_files:
    class1, class2, fold, fname = file.split('\\')[-4:]
    positive_deceptive_testing_class_dict[class1+class2].append(file)
    positive_deceptive_training_class_dict[class1+class2].append(file)

for file in positive_truthful_classifier_files:
    class1, class2, fold, fname = file.split('\\')[-4:]
    positive_truthful_testing_class_dict[class1+class2].append(file)
    positive_truthful_training_class_dict[class1+class2].append(file)


for class_type, file_list in training_class_dict.items():
    for file in file_list:
        fopen=open(os.path.join(file), "r")
        content=fopen.read()
        fopen.close()
        text_words = content.split()
        text_words = [word.lower() for word in text_words]
        word_table = str.maketrans('', '', string.punctuation)
        clean_words = [word.translate(word_table) for word in text_words]
        vocabulary.extend(clean_words)

for class_type, file_list in training_class_dict.items():
    total_docs = total_docs + len(file_list)

for class_type, file_list in training_class_dict.items():
    number_of_docs_in_class = len(file_list)
    prior_probability_count=(number_of_docs_in_class / total_docs)
    prior_probability_of_class[class_type]=(prior_probability_count)

fopen=open('nbmodel.txt', 'w')
prior_list=[]

for class_type, value in prior_probability_of_class.items():
    prior_counting2=(class_type, str(value))
    prior_list_str=(",").join(prior_counting2)
    fopen.write(prior_list_str + "\n")


for class_type, file_list in negative_deceptive_training_class_dict.items():
    for file in file_list:
        f = open(os.path.join(file), "r")
        content = f.read()
        f.close()
        text_words = content.split()
        text_words = [word.lower() for word in text_words]
        word_table = str.maketrans('', '', string.punctuation)
        clean_words1 = [word.translate(word_table) for word in text_words]
        clean_words2 = [x for x in clean_words1 if x not in stop_words]
        final_clean_words.extend(clean_words2)

    for word in vocabulary:
        if word in final_clean_words:
            final_token_count1[word]=(final_clean_words.count(word))
            final_token_count[word]=1

    for word in final_token_count1:
        final_word_count1=final_word_count1+final_token_count1.get(word)


    token_count=list(token_count)
    token_count.clear()
    final_clean_words.clear()

for class_type, file_list in negative_truthful_training_class_dict.items():
    for file in file_list:
        f = open(os.path.join(file), "r")
        content = f.read()
        f.close()
        text_words = content.split()
        text_words = [word.lower() for word in text_words]
        word_table = str.maketrans('', '', string.punctuation)
        clean_words1 = [word.translate(word_table) for word in text_words]
        clean_words2 = [x for x in clean_words1 if x not in stop_words]
        final_clean_words.extend(clean_words2)

    for word in vocabulary:
        if word in final_clean_words:
            final_token_count2[word]=(final_clean_words.count(word))
            final_token_count[word]=1

    for word in final_token_count2:
        final_word_count2=final_word_count2+final_token_count2.get(word)


    token_count=list(token_count)
    token_count.clear()
    final_clean_words.clear()

for class_type, file_list in positive_deceptive_training_class_dict.items():
    for file in file_list:
        f = open(os.path.join(file), "r")
        content = f.read()
        f.close()
        text_words = content.split()
        text_words = [word.lower() for word in text_words]
        word_table = str.maketrans('', '', string.punctuation)
        clean_words1 = [word.translate(word_table) for word in text_words]
        clean_words2 = [x for x in clean_words1 if x not in stop_words]
        final_clean_words.extend(clean_words2)

    for word in vocabulary:
        if word in final_clean_words:
            final_token_count3[word]=(final_clean_words.count(word))
            final_token_count[word]=1

    for word in final_token_count3:
        final_word_count3=final_word_count3+final_token_count3.get(word)


    token_count=list(token_count)
    token_count.clear()
    final_clean_words.clear()

for class_type, file_list in positive_truthful_training_class_dict.items():
    for file in file_list:
        f = open(os.path.join(file), "r")
        content = f.read()
        f.close()
        text_words = content.split()
        text_words = [word.lower() for word in text_words]
        word_table = str.maketrans('', '', string.punctuation)
        clean_words1 = [word.translate(word_table) for word in text_words]
        clean_words2 = [x for x in clean_words1 if x not in stop_words]
        final_clean_words.extend(clean_words2)

    for word in vocabulary:
        if word in final_clean_words:
            final_token_count4[word]=(final_clean_words.count(word))
            final_token_count[word]=1

    for word in final_token_count4:
        final_word_count4=final_word_count4+final_token_count4.get(word)


    token_count=list(token_count)
    token_count.clear()
    final_clean_words.clear()

final_word_count=final_word_count1+final_word_count2+final_word_count3+final_word_count4
total_terms=len(final_token_count)

output=""
for word in final_token_count:
    output=word
    #if word in final_token_count1:
    output+=","+str(((final_token_count1.get(word,0))+1)/(final_word_count+total_terms))+","+str(((final_token_count2.get(word,0))+1)/(final_word_count+total_terms))+","+str(((final_token_count3.get(word,0))+1)/(final_word_count+total_terms))+","+str(((final_token_count4.get(word,0))+1)/(final_word_count+total_terms))+"\n"
    fopen.write(output)
