# src/model.py  （既存の RandomForest 関数などの下に追記）

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_xgboost_model(seed: int = 42) -> XGBClassifier:
    """
    XGBoost のベースラインモデルを返します。
    必要に応じてパラメータは後で Optuna でチューニングしてください。
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
        verbosity=0
    )
def get_logistic_model(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(random_state=seed, max_iter=1000)

def get_lightgbm_model(seed: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        metric="auc",
        random_state=seed,
        n_jobs=-1
    )

def get_catboost_model(seed: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_state=seed,
        verbose=0
    )

def get_knn_model() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=5)

def get_svm_model() -> SVC:
    return SVC(probability=True, kernel='linear', random_state=42)

def get_mlp_model(seed: int = 42) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=seed)