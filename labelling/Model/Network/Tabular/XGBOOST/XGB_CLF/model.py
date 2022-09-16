from xgboost import XGBClassifier
import numpy as np
from skopt.space import Real, Integer

def returnParam(param=None):
    alg = str(param["algorithm"])
    epoch = int(param["epochs"])

    param_grid = {
        "n_estimators":Integer(1,epoch+1) if alg == "bayesian" else range(1, epoch+1),
        "max_depth":Integer(6,10) if alg == "bayesian" else range(6,10),
        "gamma":Real(0.5,1) if alg == "bayesian" else [0.5, 1],
        "colsample_bytree":Real(0.5,1) if alg == "bayesian" else [0.5, 1],
        "colsample_bylevel":Real(0.5,1) if alg == "bayesian" else [0.5, 1],
        "colsample_bynode":Real(0.5,1) if alg == "bayesian" else [0.5, 1],
        "subsample":Real(0.5,1) if alg == "bayesian" else [0.5, 1],
        "use_label_encoder":[True],
        "learning_rate":Real(0.1, 1) if alg == "bayesian" else np.arange(0.1, 1)
    }

    return param_grid
    
def createModel(param=None, saveMdlPath=None):
    
    model = XGBClassifier(
        eval_metric='mlogloss', 
        use_label_encoder=False
    )

    return model