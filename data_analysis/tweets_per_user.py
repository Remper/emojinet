import argparse
import logging
import sys
import json
import plotly.offline
import plotly.graph_objs as go

sys.path.append(sys.path[0] + "/..")

from utils.fileprovider import FileProvider
logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description='Data analysis for the ITAmoji task')
    parser.add_argument('--workdir', required=True, help='Work path')

    args = parser.parse_args()
    files = FileProvider(args.workdir)

    tweets_per_user = dict()

    with open(files.evalita, 'r', encoding="utf-8") as reader:
        for line in reader:
            line = line.rstrip()
            sample = json.loads(line)
            uid = sample["uid"]
            tid = sample["tid"]
            if tweets_per_user.get(uid):
                tweets_per_user[uid].append(tid)
            else:
                tweets_per_user[uid] = []
                tweets_per_user[uid].append(tid)

    minimum_number_of_tweets = 10000
    maximum_number_of_tweets = 1

    for user, tweets in tweets_per_user.items():
        number_of_tweets = len(tweets)
        if number_of_tweets < minimum_number_of_tweets:
            minimum_number_of_tweets = number_of_tweets
        if number_of_tweets > maximum_number_of_tweets:
            maximum_number_of_tweets = number_of_tweets

    logging.info("Number of unique users: {}".format(len(tweets_per_user)))
    logging.info("Minimum number of tweets per user: {}".format(minimum_number_of_tweets))
    logging.info("Maximum number of tweets per user: {}".format(maximum_number_of_tweets))

    logging.info("Dumping to tweets_per_user.json")
    with open("{}/tweets_per_user.json".format("data_analysis"), 'w') as outfile:
        json.dump(tweets_per_user, outfile)

    tweets_per_user_count = dict()
    for user, tweets in tweets_per_user.items():
        tweets_per_user_count["user_{}".format(str(user))] = len(tweets)


    min_frequency = 100
    dtick = 10
    X = []
    Y = []

    for element, element_count in sorted(tweets_per_user_count.items(), key=lambda kv: kv[1], reverse=False):
        if element_count > min_frequency:
            X.append(element_count)
            Y.append(element)

    plotly.offline.plot({"data": [go.Bar(orientation="h",
                                         x=X,
                                         y=Y,
                                         marker=dict(color="#3498db"))],
                         "layout": go.Layout(title="<b>{}</b>".format("Tweets per user distribution"),
                                             xaxis=dict(title="<b>{}</b>".format("tweets number"),
                                                        titlefont=dict(color="#3498db")),
                                             yaxis=dict(title="<b>{}</b>".format("user"), dtick=dtick,
                                                        titlefont=dict(color="#3498db")),
                                             margin=go.layout.Margin(l=250, r=250)
                                             )
                         },
                        filename="data_analysis/tweets_per_user_distribution.html",
                        auto_open=False)
