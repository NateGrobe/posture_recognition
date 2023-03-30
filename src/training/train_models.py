from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


# potential models to be trained and evaluated
PIPELINES = {
    # 'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    # 'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
}

def generate_models(X_train, y_train) -> dict:
    models = {}
    for mt, pipeline in PIPELINES.items():
        print(f"Training model {mt}")
        models[mt] = pipeline.fit(X_train, y_train)
        print(f"Model {mt} trained\n")


    return models

def write_model(model, type):
    with open(f"./models/{type}", "wb") as f:
        pickle.dump(model,f)


if __name__ == '__main__':
    # read in data
    df = pd.read_csv('./models/coords.csv')

    X = df.drop('class', axis=1) # features
    y = df['class'] # target value

    print("Splitting Samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    print("Samples split successfully!\n")

    print("Training models...")
    models = generate_models(X_train, y_train)

    print("\nModel training complete!\n")

    print("Writing Models to pickle files")

    for key in models:
        write_model(models[key], key)

