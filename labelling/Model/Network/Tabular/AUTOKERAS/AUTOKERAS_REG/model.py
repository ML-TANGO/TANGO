import autokeras as ak

def createModel(param=None, saveMdlPath=None):
    model = None
    if param is not None:
        loss = str(param["loss"]) if "loss" in param else 'mean_squared_error'

        model = ak.StructuredDataRegressor(
            loss=loss,
            overwrite=True,
            max_trials=int(param["max_trials"]) if 'max_trials' in param else 3,
            tuner=str(param["tunner"]) if 'tunner' in param else 'greedy',
            directory=saveMdlPath,
            project_name="tempModel"
        )

    return model