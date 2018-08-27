# given the file in input, creates a list of tokenized items and dumps it into a text file

import json
import pickle

# set file name
file = open('annotation.json', 'r')

tweets = []

index = 0

words = []
lemmas = []
postags = []

reading = False

i = 0

for line in file:
    if i % 1000 == 0:
        print(i)
    i += 1
    if "\"index\": " in line:
        current = int(line.split()[1][:-1])
        if current >= index:
            index = current
            reading = True
        else:
            tweet_list = words + lemmas + postags
            tweets.append(tweet_list)
            words = []
            lemmas = []
            postags = []
            index = 0
    if reading:
        if "\"word\": " in line:
            word = str(line.split()[1][1:-2])
        if "\"lemma\": " in line:
            lemma = str(line.split()[1][1:-2])
        if "\"pos\": " in line:
            postag = str(line.split()[1][1:-2])
            reading = False
            words.append(word)
            lemmas.append(lemma)
            postags.append(postag)
            
file.close()

with open("tweets_tokenized.txt", "wb") as fp:   #Pickling
    pickle.dump(tweets, fp)
