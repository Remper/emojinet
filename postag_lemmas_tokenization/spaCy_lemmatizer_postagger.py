# given the string of a tweet, returns the tokenized item

import spacy

nlp = spacy.load('it_core_news_sm')

def tokenize(tweet):

    tokenized_tweet = []

    nlp_tweet = nlp(tweet)

    for word in nlp_tweet:
        tokenized_tweet.append(word.text)
    for word in nlp_tweet:
        tokenized_tweet.append(word.pos_)
    for word in nlp_tweet:
        tokenized_tweet.append(word.lemma_)

    return tokenized_tweet
