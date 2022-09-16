import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pandas as pd

from qhoptim.pyt import QHAdam

import traceback
# Model 폴더 찾기 위한 방법
pathA = os.path.join(os.path.dirname(__file__), os.path.pardir, "../../../")
pathB = os.path.join(os.path.join(pathA))
basePath = os.path.abspath(pathB)
# Model Path 등록

sys.path.append(basePath)

from Common.Logger.Logger import logger
from Common.Utils.Utils import getConfig
from Common.Process.Process import prcErrorData, prcSendData, prcGetArgs, prcLogData
from DatasetLib import DatasetLib

from Network.Tabular.PYTORCH.NODE import lib


log = logger("log")

# logs폴더 삭제
import shutil
try:
    shutil.rmtree(os.path.join(basePath, 'Network/Tabular/PYTORCH/NODE/logs'))
except:
    pass


if __name__ == "__main__":
    try:
        param = json.loads(sys.argv[1])
        # param = '{"INPUT_DATA":{"FILE_PATH":"/Users/parksangmin/Downloads/","TRAIN_FILE_NAMES":["kddcup99_csv.csv"],"DELIMITER":",","SPLIT_RATIO":0.25,"LABEL_COLUMN_NAME":"label","MAPING_INFO":{"aaa":"aaa"}},"SERVER_PARAM":{"AI_CD":"eeeee"},"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/TF/TabNetCLF","HYPER_PARAM":{"num_decision_steps":7,"relaxation_factor":1.5,"sparsity_coefficient":1e-05,"batch_momentum":0.98},"MODEL_PARAM":{"BATCH_SIZE":10,"EPOCH":10}}}'
        # param = json.loads(param)
        # prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Trainer Run"}))

        dataLib = DatasetLib.DatasetLib()
        param = dataLib.setParams(param)

        device = param["GPU_IDX"] if "GPU_IDX" in param else 'cpu'

        xTrain, yTrain, xTest, yTest, colNames, labelNames, labels, output = dataLib.getDataFrame(param)

        xTrain = xTrain.values
        yTrain = yTrain.values
        xTest = xTest.values
        yTest = yTest.values

        data = dict(
            X_train=xTrain.astype(np.float32),
            y_train=yTrain.astype(np.int64),
            X_valid=xTest.astype(np.float32),
            y_valid=yTest.astype(np.int64),
            X_test=xTest.astype(np.float32),
            y_test=yTest.astype(np.int64)
        )
        mu, std = data['y_train'].mean(), data['y_train'].std()
        normalize = lambda x: ((x - mu) / std).astype(np.float32)
        data['y_train'], data['y_valid'], data['y_test'] = map(normalize, [data['y_train'], data['y_valid'], data['y_test']])

        # mu, std = trainDs.y.mean(), trainDs.y_train.std()
        # normalize = lambda x: ((x - mu) / std).astype(np.float32)

        layerDim = int(param["LAYER_DIM"]) if "LAYER_DIM" in param else 128
        numLayers = int(param["NUM_LAYERS"]) if "NUM_LAYERS" in param else 8
        treeDim = int(param["TREE_DIM"]) if "TREE_DIM" in param else 3
        depth = int(param["DEPTH"]) if "DEPTH" in param else 6
        epochs = int(param["EPOCH"]) if "EPOCH" in param else 100
        batchSize = int(param["BATCH_SIZE"]) if "BATCH_SIZE" in param else 128

        cuda = False
        device = 'cpu' if cuda is False else 'cuda'

        model = nn.Sequential(
            lib.DenseBlock(
                xTrain.shape[1],
                layer_dim=layerDim,
                num_layers=numLayers,
                tree_dim=treeDim,
                depth=depth,
                flatten_output=False,
                choice_function=lib.entmax15,
                bin_function=lib.entmoid15
            ),
            lib.Lambda(lambda x: x[..., 0].mean(dim=-1))
        ).to(device)

        optimizer = QHAdam
        optimizer_params = {
            'nus': (0.7, 1.0),
            'betas': (0.95, 0.998)
        }

        trainer = lib.Trainer(
            model=model,
            loss_function=F.mse_loss,
            experiment_name="NODE",
            warm_start=False,
            Optimizer=optimizer,
            optimizer_params=optimizer_params,
            verbose=True,
            n_last_checkpoints=5
        )

        loss_history, mse_history = [], []
        best_mse = float('inf')
        early_stopping_rounds = 5000
        report_frequency = 1

        for epoch, batch in enumerate(lib.iterate_minibatches(
                data['X_train'], data['y_train'], 
                batch_size=int(param["batch_size"]) if "batch_size" in param else 128, 
                shuffle=True, 
                epochs=int(param["epochs"]) if "epochs" in param else 100)):

            metrics = trainer.train_on_batch(*batch, device=device)
            print("epoch : {}, loss : {}".format(epoch + 1, metrics['loss']))

            loss_history.append(metrics['loss'])

        # mse 계산
        # mse = trainer.evaluate_mse(
        #         data['X_valid'],
        #         data['y_valid'],
        #         device=device,
        #         batch_size=batchSize
        #     )

        #     print("epoch : {}, loss : {}, MSE : {}".format(epoch + 1, metrics['loss'], mse))

            # if trainer.step % report_frequency == 0:
            #     trainer.save_checkpoint()
            #     trainer.average_checkpoints(out_tag='avg')
            #     trainer.load_checkpoint(tag='avg')
                

            #     if mse < best_mse:
            #         best_mse = mse
            #         best_step_mse = trainer.step
            #         trainer.save_checkpoint(tag='best_mse')

            #     mse_history.append(mse)
                
                # trainer.load_checkpoint()  # last
                # trainer.remove_old_temp_checkpoints()

    except Exception as e:
        # log.error(e)
        # print(e)
        traceback.print_exc()

    finally:
        sys.exit()
