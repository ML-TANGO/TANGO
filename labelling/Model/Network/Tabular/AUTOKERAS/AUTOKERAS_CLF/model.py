import autokeras as ak

def createModel(param=None, saveMdlPath=None):
    model = None
    colType = dict()
    if param is not None:
        loss = str(param["loss"]) if "loss" in param else None
        if loss != 'categorical_crossentropy' or loss != 'binary_crossentropy' or loss is None:
            loss = 'categorical_crossentropy' if len(param['LABELS']) > 1 else 'binary_crossentropy'

        for columns in param["COLUMNS"]:
            colType[columns] = "numerical"

        model = ak.StructuredDataClassifier(
            column_names=param["COLUMNS"],
            column_types=colType,
            num_classes=len(param["LABELS"]),
            multi_label=True if len(param['LABELS']) > 2 else False,
            loss=loss,
            overwrite=True,
            max_trials=int(param["max_trials"]) if 'max_trials' in param else 3,
            tuner=str(param["tunner"]) if 'tunner' in param else 'greedy',
            directory=saveMdlPath,
            project_name="tempModel"
        )

    return model