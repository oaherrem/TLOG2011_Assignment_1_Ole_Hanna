from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter= 1000,
            class_weight= "balanced"
        ))
    ])
    model.fit(X_train, y_train)
    return model

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
