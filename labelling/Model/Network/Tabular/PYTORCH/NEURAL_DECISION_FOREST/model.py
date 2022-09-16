from Network.Tabular.PYTORCH.NEURAL_DECISION_FOREST import ndf
import json
from Common.Logger.Logger import logger

import os
log = logger("log")


def createModel(param=None, saveMdlPath=None):
    if param is None:
        with open(os.path.join(saveMdlPath, "param.json"), "r") as f: 
            jsonData = json.load(f)

        featureLayer = ndf.CustomSetFeatureLayer(
            featureLen=len(jsonData["COLUMNS"])
        )
        forest = ndf.Forest(
            n_tree=5,
            tree_depth=3,
            n_in_feature=featureLayer.get_out_feature_size(),
            tree_feature_rate=0.5,
            jointly_training=True,
            n_class=len(jsonData["LABELS"])
        )
        model = ndf.NeuralDecisionForest(featureLayer, forest)

    else:
        # create NDF MODEL
        featureLayer = ndf.CustomSetFeatureLayer(
            dropout_rate=float(param["feat_dropout"]) if "feat_dropout" in param else 0.3,
            shallow=True,
            featureLen=len(param["COLUMNS"]) if len(param["COLUMNS"]) > 0 else 5
        )
        forest = ndf.Forest(
            n_tree=int(param["n_tree"]) if "n_tree" in param else 5,
            tree_depth=int(param["tree_depth"]) if "tree_depth" in param else 3,
            n_in_feature=featureLayer.get_out_feature_size(),
            tree_feature_rate=float(param["tree_feature_rate"]) if "tree_feature_rate" in param else 0.5,
            n_class=len(param["LABELS"]),
            jointly_training=True,
        )

        model = ndf.NeuralDecisionForest(featureLayer, forest)

    return model