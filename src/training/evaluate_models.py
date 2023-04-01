'''
    Uses the models saved as pickle files and evaluates them on data
    that was not used for the training or test sets.

    The purpose of this is to evaluate how well the models can generalize in
    order to determine the best model for this task.
'''

import pickle
import pandas as pd
import pathlib
from create_csv import create_csv, get_image_list
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_test_vals():
    c_path = pathlib.Path().resolve()
    img_list = get_image_list(c_path)
    create_csv(f"{c_path}/evaluation_data/coords.csv", img_list)

    df = pd.read_csv('./evaluation_data/coords.csv')

    X = df.drop('class', axis=1)
    y = df['class']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    return X_test, y_test

def test_model(model, x, y):
    yhat = model.predict(x)
    return accuracy_score(y, yhat)

def evaluate_models(x, y):
    rf_model, rc_model = None, None
    with open('./models/rf.pkl') as f:
        rf_model = pickle.load(f)

    with open('./models/rc.pkl') as f:
        rc_model = pickle.load(f)

    print("Accuracy of Random Forest:")
    rf_acc = test_model(rf_model, x, y)
    print(rf_acc)

    print("Accuracy of Ridge Classification:")
    rf_acc = test_model(rf_model, x, y)
    print(rf_acc)

if __name__ == '__main__':
    x, y = get_test_vals()
    evaluate_models(x, y)