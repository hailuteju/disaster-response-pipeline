import os
import sys
import re
import pandas as pd
import pickle
from pathlib import Path
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (confusion_matrix, classification_report)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer)

stop_words = stopwords.words("english")


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = (df.drop(columns=['id', 'message', 'original', 'genre'])
                      .columns).tolist()

    return X, Y, category_names


def tokenize(text):
    regex_url = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    text = regex_url.sub("urlplaceholder", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip()
                    for tok in tokens if tok not in stop_words]

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # specify parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    # create grid search object
    cv = GridSearchCV(pipeline,
                      param_grid=parameters,
                      n_jobs=-1,
                      verbose=2, )

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    score = model.score(X_test, Y_test)
    print(f"Model score: {score}", end="\n\n")
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f"{category}:")
        print(classification_report(Y_test[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    # save the model
    pickle.dump(model, open(model_filepath, 'wb'))


def load_model(model_filepath):
    # load the model from disk
    model = pickle.load(open(model_filepath, 'rb'))

    return model


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument \nand the filepath of the pickle file to ' \
              'save the model to as the second argument. \nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
