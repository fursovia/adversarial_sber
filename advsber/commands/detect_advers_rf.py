import pandas as pd
import numpy as np
import jsonlines
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
import typer

def load_jsonlines(path: str):
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data

def main(dataset_name: str, attack_name: str, subst_name: str, targ_name: str):
    train = load_jsonlines("../experiments/attacks/" + dataset_name + "/targ_" + targ_name + "/subst_" + subst_name + "/" + attack_name + "/train_adv_detection_dataset.jsonl")
    test = load_jsonlines("../experiments/attacks/" + dataset_name + "/targ_" + targ_name + "/subst_" + subst_name + "/" + attack_name + "/test_adv_detection_dataset.jsonl")
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    special_tokens = ['@@MASK@@', '@@UNKNOWN@@', '@@PADDING@@', '<START>', '<END>']

    for dataset in [train, test]:
        for i in range(dataset.shape[0]):
            for t in range(len(dataset.transactions[i])):
                if dataset.transactions[i][t] in special_tokens:
                    dataset.drop(i, axis=0, inplace=True)
        dataset.reset_index(inplace=True)

    vectorizer0 = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    corpus_full = list( str(train.transactions[i]) for i in range(train.shape[0])) + list( str(test.transactions[i]) for i in range(test.shape[0]))
    vocabulary = vectorizer0.fit(corpus_full).vocabulary_

    vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b', vocabulary=vocabulary)
    corpus_train = list( str(train.transactions[i]) for i in range(train.shape[0]))
    corpus_test = list( str(test.transactions[i]) for i in range(test.shape[0]))

    X_train = vectorizer.fit_transform(corpus_train)
    X_train.toarray()
    X_test = vectorizer.fit_transform(corpus_test)
    X_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    X_train = transformer.fit_transform(X_train).toarray()
    X_test = transformer.fit_transform(X_test).toarray()

    y_train = np.array(train.label)
    y_test = np.array(test.label)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('accuracy=', accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    typer.run(main)






























