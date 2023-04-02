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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# generates test values from photos
def get_test_vals():
    c_path = pathlib.Path().resolve()
    img_list = get_image_list(f"{c_path}/evaluation_data")
    # create_csv(f"{c_path}/evaluation_data/coords.csv", img_list)

    df = pd.read_csv('./evaluation_data/coords.csv')

    X = df.drop('class', axis=1)
    y = df['class']

    # use most of the data as test data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    return X_test, y_test

# tests the predict model for accuracy
def test_accuracy(model, x, y) -> float:
    yhat = model.predict(x)
    return accuracy_score(y, yhat)

def test_precision(model, x, y) -> float:
    yhat = model.predict(x)
    return precision_score(y, yhat, average='micro')

def test_recall(model, x, y) -> float:
    yhat = model.predict(x)
    return recall_score(y, yhat, average='micro')

def test_f1(model, x, y) -> float:
    yhat = model.predict(x)
    return f1_score(y, yhat, average='micro')

# prints results
def evaluate_models(x, y):
    rf_model, rc_model, gb_model, lr_model = None, None, None, None
    with open('./models/rf.pkl', "rb") as f:
        rf_model = pickle.load(f)

    with open('./models/gb.pkl', "rb") as f:
        gb_model = pickle.load(f)

    with open('./models/rc.pkl', "rb") as f:
        rc_model = pickle.load(f)

    with open('./models/lr.pkl', "rb") as f:
        lr_model = pickle.load(f)


    print("Accuracy:")
    rf_acc = test_accuracy(rf_model, x, y)
    print(f"Random forest: {rf_acc}")

    rc_acc = test_accuracy(rc_model, x, y)
    print(f"Ridge Classification: {rc_acc}")

    gb_acc = test_accuracy(gb_model, x, y)
    print(f"Gradient Boosting: {gb_acc}")

    lr_acc = test_accuracy(lr_model, x, y)
    print(f"Logistic Regression: {lr_acc}")

    print("\nPrecision")
    print(f"Random Forest: {test_precision(rf_model, x, y)}")
    print(f"Ridge Classification: {test_precision(rc_model, x, y)}")
    print(f"Gradient Boosting: {test_precision(gb_model, x, y)}")
    print(f"Logistic Regression: {test_precision(lr_model, x, y)}")

    print("\nRecall:")
    print(f"Random Forest: {test_recall(rf_model, x, y)}")
    print(f"Ridge Classification: {test_recall(rc_model, x, y)}")
    print(f"Gradient Boosting: {test_recall(gb_model, x, y)}")
    print(f"Logistic Regression: {test_recall(lr_model, x, y)}")

    print("\nF1:")
    print(f"Random Forest: {test_f1(rf_model, x, y)}")
    print(f"Ridge Classification: {test_f1(rc_model, x, y)}")
    print(f"Gradient Boosting: {test_f1(gb_model, x, y)}")
    print(f"Logistic Regression: {test_f1(lr_model, x, y)}")


if __name__ == '__main__':
    x, y = get_test_vals()
    evaluate_models(x, y)