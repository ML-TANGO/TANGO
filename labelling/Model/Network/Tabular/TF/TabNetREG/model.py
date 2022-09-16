from Network.Tabular.TF.TabNetCLF import tabnet
import tensorflow as tf
import json
import os

def createModel(param=None, saveMdlPath=None):
    if param is None:
        with open(os.path.join(saveMdlPath, "param.json"), "r") as f:
            jsonData = json.load(f)

        featureColumns = []
        for colName in jsonData["COLUMNS"]:
            featureColumns.append(tf.feature_column.numeric_column(colName))

        model = tabnet.TabNetRegressor(
            featureColumns,
            num_regressors=jsonData["num_regressors"],
            num_features=len(jsonData["COLUMNS"]),
            feature_dim=len(jsonData["COLUMNS"]) * 2,
            output_dim=len(jsonData["COLUMNS"]),
            dynamic=True
        )

    else:
        featureColumns = []
        for colName in param["COLUMNS"]:
            featureColumns.append(tf.feature_column.numeric_column(colName))

        model = tabnet.TabNetRegressor(
            featureColumns,
            num_regressors=int(param["num_regressors"]) if "num_regressors" in param else 3,
            num_features=len(param["COLUMNS"]),
            feature_dim=len(param["COLUMNS"]) * 2,
            output_dim=len(param["COLUMNS"]),
            num_decision_steps=int(param["num_decision_steps"]) if "num_decision_steps" in param else len(param["COLUMNS"]),
            relaxation_factor=float(param["relaxation_factor"]) if "relaxation_factor" in param else 1.5,
            sparsity_coefficient=float(param["sparsity_coefficient"]) if "sparsity_coefficient" in param else 1e-5,
            batch_momentum=float(param["batch_momentum"]) if "batch_momentum" in param else 0.98,
            virtual_batch_size=int(param["virtual_batch_size"]) if "virtual_batch_size" in param and param["virtual_batch_size"] != "None" else None,
            norm_type='group',
            num_groups=int(param["num_groups"]) if "num_groups" in param else 1
        )

    return model