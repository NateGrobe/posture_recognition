from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


# potential models to be trained and evaluated
PIPELINES = {
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
}

def generate_models(X_train, y_train) -> dict:
    models = {}
    for mt, pipeline in PIPELINES.items():
        models[mt] = pipeline.fit(X_train, y_train)

    return models

if __name__ == '__main__':
    # read in data
    df = pd.read('../models/coords.csv')

    X = df.drop('class', axis=1) # features
    y = df['class'] # target value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    models = generate_models()


    
