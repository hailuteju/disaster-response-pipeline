# from . import app
import json
import io
# import os
import re
# import pickle
import joblib
import nltk
import pandas as pd
import plotly
import dvc.api
# from pathlib import Path
from flask import Flask
from flask import render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sklearn.base import (BaseEstimator, TransformerMixin)
# import plotly.express as px
# from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download([
    'punkt', 'wordnet', 'words', 'stopwords', 'averaged_perceptron_tagger'
])

stop_words = stopwords.words("english")

app = Flask(__name__)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0: continue
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    regex_url = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    text = regex_url.sub("urlplaceholder", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip()
                    for tok in tokens if tok not in stop_words]

    return clean_tokens


# load model
with dvc.api.open(
        "models/model.pkl",
        repo='https://github.com/hailuteju/disaster-response-pipeline',
        mode="rb") as f:
    model = joblib.load(f)

# load data
msgs = dvc.api.read(
    'data/messages.csv',
    repo='https://github.com/hailuteju/disaster-response-pipeline'
)
messages_csv = io.StringIO(msgs)
messages_df = pd.read_csv(messages_csv)

col_means = messages_df[messages_df.columns[4:]].mean().sort_values(
    ascending=False)
data = pd.DataFrame(col_means).reset_index()
data.columns = ['Category', 'Mean']


# # load data
# root_path = Path(os.path.dirname(__file__)).parent.absolute()
# db_path = f"{root_path}/data/DisasterResponse.db"
# engine = create_engine(f'sqlite:///{db_path}')
# messages_df = pd.read_sql_table('messages', engine)
#
# col_means = messages_df[messages_df.columns[4:]].mean().sort_values(
#     ascending=False)
# data = pd.DataFrame(col_means).reset_index()
# data.columns = ['Category', 'Mean']
#
# # load model
# model = joblib.load(f"{root_path}/models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data for your own visuals
    genre_counts = messages_df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(x=genre_names, y=genre_counts)
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'xaxis': {'title': "Genre"},
                'yaxis': {'title': "Count"}
            }
        },
        {
            'data': [Bar(x=data.Category, y=data.Mean)],
            'layout': {
                'title': 'Average Frequency of Message Categories'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', default='')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(
        zip(messages_df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
