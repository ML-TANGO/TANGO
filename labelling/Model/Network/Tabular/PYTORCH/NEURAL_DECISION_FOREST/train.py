import json
import os
import sys
import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import time

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
from Output.Output import sendMsg

from DatasetLib import DatasetLib

from Network.Tabular.PYTORCH.NEURAL_DECISION_FOREST import predict, model

log = logger("log")

if __name__ == "__main__":
    try:
        # param = json.loads(prcGetArgs(0))
        # data = '{"INPUT_DATA":{"FILE_PATH":"/Users/parksangmin/Downloads/","TRAIN_FILE_NAMES":["kddcup99_csv.csv"],"DELIMITER":",","SPLIT_RATIO":0.25,"LABEL_COLUMN_NAME":"label","MAPING_INFO":{"aaa":"aaa"}},"SERVER_PARAM":{"AI_CD":"TD000001","SRV_IP":"192.168.0.2","SRV_PORT":10235,"TRAIN_RESULT_URL":"trainBinLog","TRAIN_STATE_URL":"trainStatusLog"},"MODEL_INFO":{"MODEL_PATH":"Network/Tabular/TF/TabNetCLF","HYPER_PARAM":{"num_decision_steps":7,"relaxation_factor":1.5,"sparsity_coefficient":1e-05,"batch_momentum":0.98},"MODEL_PARAM":{"BATCH_SIZE":10,"EPOCH":10}}}'
        param = json.loads(sys.argv[1])
        # param = json.loads(param)
        # prcSendData(__file__, json.dumps({"TYPE": "LOG", "DATA": "Trainer Run"}))

        # set Param
        dataLib = DatasetLib.DatasetLib()
        param = dataLib.setParams(param)

        saveMdlPath = os.path.join(param["SERVER_PARAM"]["AI_PATH"], str(param["MDL_IDX"]))
        trainStart = dataLib.setStatusOutput(param, "train start", os.getpid(), True)
        _ = sendMsg(trainStart["SRV_ADDR"], trainStart["SEND_DATA"])

        if not os.path.isdir(saveMdlPath):
            os.makedirs(saveMdlPath, exist_ok=True)

        # get data
        trainDs, testDs, colNames, param["LABEL_COLUMN_NAME"], labels, output = dataLib.getTorchDataset(param)

        param["COLUMNS"] = colNames
        param["LABELS"] = labels
        with open(os.path.join(saveMdlPath, "param.json"), "w") as f:
            json.dump(param, f)

        # create NDF MODEL
        try:
            model = model.createModel(param=param, saveMdlPath=saveMdlPath)

        except Exception as e:
            trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
            log.error(str(e))
            log.error(traceback.format_exc())
            _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])
            sys.exit()

        # set optimizer (adam)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=float(param["learning_rate"]) if "learning_rate" in param else 1e-3, weight_decay=1e-5)

        epochs = int(param["epochs"]) if "epochs" in param else 100
        batchSize = int(param["batch_size"]) if "batch_size" in param else 128
        # cuda = True if torch.cuda.is_available() else False
        cuda = False
        # cuda = True if torch.device("cuda" if torch.cuda.is_available() else "cpu") else False

        model = model.cuda() if cuda else model.cpu()
        
        best_loss = 999999999999
        best_acc = -999999999999
        earlyStopCnt = 0
        fstEpochStartTime = 0
        fstEpochEndTime = 0
        # train Model
        for epoch in range(1, epochs + 1):
            cls_onehot = torch.eye(len(param["LABELS"]))
            feat_batches = []
            target_batches = []
            train_loader = torch.utils.data.DataLoader(
                trainDs,
                batch_size=batchSize,
                shuffle=True
            )
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(train_loader):
                    target = target.type(torch.long)
                    if cuda:
                        data, target, cls_onehot = data.cuda(), target.cuda(), cls_onehot.cuda()
                    data = Variable(data)
                    # Get feats
                    feats = model.feature_layer(data)
                    feats = feats.view(feats.size()[0], -1)
                    feat_batches.append(feats)
                    target_batches.append(cls_onehot[target])

                # Update \Pi for each tree
                for tree in model.forest.trees:
                    mu_batches = []
                    for feats in feat_batches:
                        mu = tree(feats)  # [batch_size,n_leaf]
                        mu_batches.append(mu)
                    for _ in range(20):
                        new_pi = torch.zeros((tree.n_leaf, tree.n_class))  # Tensor [n_leaf,n_class]
                        if cuda:
                            new_pi = new_pi.cuda()
                        for mu, target in zip(mu_batches, target_batches):
                            pi = tree.get_pi()  # [n_leaf,n_class]
                            prob = tree.cal_prob(mu, pi)  # [batch_size,n_class]

                            # Variable to Tensor
                            pi = pi.data
                            prob = prob.data
                            mu = mu.data

                            _target = target.unsqueeze(1)  # [batch_size,1,n_class]
                            _pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
                            _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
                            _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)  # [batch_size,1,n_class]

                            _new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob  # [batch_size,n_leaf,n_class]
                            new_pi += torch.sum(_new_pi, dim=0)

                        new_pi = F.softmax(Variable(new_pi), dim=1).data
                        tree.update_pi(new_pi)

            # Update \Theta

            if epoch == 1:
                fstEpochStartTime = time.time()
            model.train()

            loss = 0
            train_loader = torch.utils.data.DataLoader(trainDs, batch_size=batchSize, shuffle=True)
            for batch_idx, (data, target) in enumerate(train_loader):
                target = target.type(torch.long)
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(torch.log(output), target)
                loss.backward()
                optimizer.step()

            # Eval
            model.eval()
            test_loss = 0
            correct = 0
            test_loader = torch.utils.data.DataLoader(testDs, batch_size=batchSize, shuffle=True)
            with torch.no_grad():
                for data, target in test_loader:
                    target = target.type(torch.long)
                    if cuda:
                        data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)
                    output = model(data)
                    test_loss += F.nll_loss(torch.log(output), target, size_average=False).item()  # sum up batch loss
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                test_loss /= len(test_loader.dataset)
            if epoch == 1:
                fstEpochEndTime = time.time() - fstEpochStartTime
            remTime = fstEpochEndTime * (epochs - epoch)

            test_acc = float(correct / len(test_loader.dataset))
            output = {
                "EPOCH": epoch,
                "LOSS": test_loss,
                "ACCURACY": test_acc,
                "MDL_IDX": param["MDL_IDX"],
                "MODEL_NAME": param["MODEL_NAME"],
                "SAVE_MDL_PATH": saveMdlPath,
                "REMANING_TIME": remTime
            }
            log.info(output)
            result = dataLib.setTrainOutput(param, output)
            _ = sendMsg(result["SRV_ADDR"], result["SEND_DATA"])

            clfMaxModelList = ["accuracy"]
            clfMinModelList = ["loss"]

            if str(param["mode"]) == "auto":
                if str(param["monitor"]) in clfMinModelList:
                    if test_loss < best_loss:
                        best_loss = test_loss
                        torch.save(model.state_dict(), os.path.join(saveMdlPath, 'weight.pt'))
                        if str(param["early_stopping"]) == "TRUE":
                            earlyStopCnt = 0
                    else:
                        if str(param["early_stopping"]) == "TRUE":
                            earlyStopCnt += 1

                elif str(param["monitor"]) in clfMaxModelList:
                    if test_acc > best_acc:
                        best_acc = test_acc                  
                        torch.save(model.state_dict(), os.path.join(saveMdlPath, 'weight.pt'))
                        if str(param["early_stopping"]) == "TRUE":
                            earlyStopCnt = 0
                    else:
                        if str(param["early_stopping"]) == "TRUE":
                            earlyStopCnt += 1

            elif str(param["mode"]) == "min":
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), os.path.join(saveMdlPath, 'weight.pt'))
                    if str(param["early_stopping"]) == "TRUE":
                        earlyStopCnt = 0
                else:
                    if str(param["early_stopping"]) == "TRUE":
                        earlyStopCnt += 1

            elif str(param["mode"]) == "max":
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), os.path.join(saveMdlPath, 'weight.pt'))
                    if str(param["early_stopping"]) == "TRUE":
                        earlyStopCnt = 0
                else:
                    if str(param["early_stopping"]) == "TRUE":
                        earlyStopCnt += 1

            if earlyStopCnt >= 10:
                break

        # 그래프 표시용 Predict
        score, graph = predict.runPredict(
            model,
            param=param,
            testDs=testDs,
            cuda=cuda,
            classes=labels,
            flag=1
        )
        output = {
            "SCORE_INFO": {
                "AI_ACC": score
            },
            "GRAPH_INFO": graph
        }

        predictData = dataLib.setPredictOutput(param, output)
        _ = sendMsg(predictData["SRV_ADDR"], predictData["SEND_DATA"])

        trainDone = dataLib.setStatusOutput(param, "train done", os.getpid(), False)
        _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        trainDone = dataLib.setStatusOutput(param, str(e), os.getpid(), False)
        _ = sendMsg(trainDone["SRV_ADDR"], trainDone["SEND_DATA"])

    finally:
        sys.exit()
