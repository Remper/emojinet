import argparse
import logging
import sys
import json
import plotly.offline
import plotly.graph_objs as go

sys.path.append(sys.path[0] + "/..")

from utils.fileprovider import FileProvider
from preprocessing.reader import EvalitaDatasetReader
from nltk.tokenize import TweetTokenizer

logging.getLogger().setLevel(logging.INFO)


def plot_distribution(dictionary, title, x_axis_title, y_axis_title, min_frequency, dtick, color, output_path):
    logging.info("Plotting {}".format(title))
    min_frequency = min_frequency
    X = []
    Y = []

    for element, element_count in sorted(dictionary.items(), key=lambda kv: kv[1], reverse=False):
        if element_count > min_frequency:
            X.append(element_count)
            Y.append(element)

    plotly.offline.plot({"data": [go.Bar(orientation="h",
                                         x=X,
                                         y=Y,
                                         marker=dict(color=color))],
                         "layout": go.Layout(title="<b>{}</b>".format(title),
                                             xaxis=dict(title="<b>{}</b>".format(x_axis_title),
                                                        titlefont=dict(color=color)),
                                             yaxis=dict(title="<b>{}</b>".format(y_axis_title), dtick=dtick,
                                                        titlefont=dict(color=color)),
                                             margin=go.layout.Margin(l=250, r=250)
                                             )
                         },
                        filename=output_path,
                        auto_open=False)


if __name__ == '__main__':
    """##### Parameter parsing"""

    parser = argparse.ArgumentParser(description='Data analysis for the ITAmoji task')
    parser.add_argument('--workdir', required=True, help='Work path')

    args = parser.parse_args()
    files = FileProvider(args.workdir)

    logging.info("Loading txt_2_emoji.json file")
    with open("{}/{}".format("data_analysis", "txt_2_emoji.json"), 'r') as txt_2_emoji_file:
        txt_2_emoji = json.load(txt_2_emoji_file)

    logging.info("Loading idx_2_emoji.json file")
    with open("{}/{}".format("data_analysis", "idx_2_emoji.json"), 'r') as idx_2_emoji_file:
        idx_2_emoji = json.load(idx_2_emoji_file)

    logging.info("Starting data analysis with parameters: {0}".format(vars(args)))
    raw_train = EvalitaDatasetReader(files.evalita)

    train_token_dict = dict()
    train_hashtag_dict = dict()
    train_mention_dict = dict()
    train_url_dict = dict()
    train_label_dict = dict()

    tweet_tokenizer = TweetTokenizer()

    logging.info("Computing counts for train set")
    for train_tweet_text, train_tweet_label in zip(raw_train.X, raw_train.Y):
        # tokens
        for token in tweet_tokenizer.tokenize(train_tweet_text.lower()):
            # general token
            train_token_dict[token] = train_token_dict[token] + 1 if train_token_dict.get(token) else 1
            if token.startswith("#"): # hashtag token
                train_hashtag_dict[token] = train_hashtag_dict[token] + 1 if train_hashtag_dict.get(token) else 1
            if token.startswith("@"): # mention token
                train_mention_dict[token] = train_mention_dict[token] + 1 if train_mention_dict.get(token) else 1
            if token.startswith("http"): # url token
                train_url_dict[token] = train_url_dict[token] + 1 if train_url_dict.get(token) else 1

        # labels
        train_label_dict[train_tweet_label] = train_label_dict[train_tweet_label] + 1 if train_label_dict.get(train_tweet_label) else 1

    with open("data_analysis/data_analysis.txt", 'w') as data_analysis_output:

        total_number_of_tokens = sum([count for token, count in train_token_dict.items()])
        total_number_of_unique_tokens = len(train_token_dict)
        logging.info("Total number of tokens: {}".format(total_number_of_tokens))
        data_analysis_output.write("Total number of tokens: {}\n".format(total_number_of_tokens))
        logging.info("Total number of unique tokens: {}".format(total_number_of_unique_tokens))
        data_analysis_output.write("Total number of unique tokens: {}\n".format(total_number_of_unique_tokens))

        total_number_of_hashtags = sum([count for token, count in train_hashtag_dict.items()])
        total_number_of_unique_hashtags = len(train_hashtag_dict)
        logging.info("Total number of hashtags: {}".format(total_number_of_hashtags))
        data_analysis_output.write("Total number of hashtags: {}\n".format(total_number_of_hashtags))
        logging.info("Total number of unique hashtags: {}".format(total_number_of_unique_hashtags))
        data_analysis_output.write("Total number of unique hashtags: {}\n".format(total_number_of_unique_hashtags))

        total_number_of_mentions = sum([count for token, count in train_mention_dict.items()])
        total_number_of_unique_mentions = len(train_mention_dict)
        logging.info("Total number of mentions: {}".format(total_number_of_mentions))
        data_analysis_output.write("Total number of mentions: {}\n".format(total_number_of_mentions))
        logging.info("Total number of unique mentions: {}".format(total_number_of_unique_mentions))
        data_analysis_output.write("Total number of unique mentions: {}\n".format(total_number_of_unique_mentions))

        total_number_of_urls = sum([count for token, count in train_url_dict.items()])
        total_number_of_unique_urls = len(train_url_dict)
        logging.info("Total number of URLs: {}".format(total_number_of_urls))
        data_analysis_output.write("Total number of URLs: {}\n".format(total_number_of_urls))
        logging.info("Total number of unique URLs: {}".format(total_number_of_unique_urls))
        data_analysis_output.write("Total number of unique URLs: {}\n".format(total_number_of_unique_urls))

        total_number_of_labels = sum([count for token, count in train_label_dict.items()])
        total_number_of_unique_labels = len(train_label_dict)
        logging.info("Total number of labels: {}".format(total_number_of_labels))
        data_analysis_output.write("Total number of labels: {}\n".format(total_number_of_labels))
        logging.info("Total number of unique labels: {}".format(total_number_of_unique_labels))
        data_analysis_output.write("Total number of unique labels: {}\n".format(total_number_of_unique_labels))

    plot_distribution(train_token_dict, "token distribution", "frequency", "token", 2000, 2, "#3498db", "data_analysis/token_distribution.html")
    plot_distribution(train_hashtag_dict, "hashtag distribution", "frequency", "hashtag", 250, 2, "#3498db", "data_analysis/hashtag_distribution.html")
    plot_distribution(train_mention_dict, "mention distribution", "frequency", "mention", 150, 2, "#3498db", "data_analysis/mention_distribution.html")
    plot_distribution(train_url_dict, "URL distribution", "frequency", "URL", 5, 2, "#3498db", "data_analysis/url_distribution.html")

    logging.info("Plotting label distribution")
    min_frequency = 0
    X_label = []
    Y_label = []

    for label, label_count in sorted(train_label_dict.items(), key=lambda kv: kv[1], reverse=True):
        if label_count > min_frequency:
            X_label.append(label_count)
            Y_label.append(idx_2_emoji[str(label)])

    plotly.offline.plot({"data": [go.Bar(orientation="h",
                                         x=X_label,
                                         y=Y_label,
                                         marker=dict(color="#3498db"))],
                         "layout": go.Layout(title="<b>label distribution</b>",
                                             xaxis=dict(title="<b>label</b>",
                                                        titlefont=dict(color="#3498db")),
                                             yaxis=dict(title="<b>frequency</b>", dtick=1,
                                                        titlefont=dict(color="#3498db")),
                                             margin=go.layout.Margin(l=250, r=250)
                                             )
                         },
                        filename="data_analysis/label_distribution.html",
                        auto_open=False)