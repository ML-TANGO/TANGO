# -*- coding:utf-8 -*-
'''
AutoML Network 생성 스크립트
1. autoML(shape, maxTrial, autoKerasPath) : AutoKeras 기반 AutoML 모델 생성
  shape = TrainX shape, maxTraial = AutoML 시도 횟수, autoKerasPath : 모델 저장 위치
'''
import autokeras as ak


def autoML(shape, maxTrial, autoKerasPath):
    # inputNode = ak.ImageInput(shape=shape)
    # outputNode = ak.Normalization()(inputNode)
    # outputNode = ak.ImageAugmentation()(outputNode)
    # outputNode1 = ak.ConvBlock()(outputNode)
    # outputNode2 = ak.XceptionBlock()(outputNode)

    # outputNode = ak.Merge()([outputNode1 + outputNode2])
    # outputNodeC = ak.ClassificationHead()(outputNode)
    # classifier = ak.AutoModel(inputs=inputNode, outputs=[outputNodeC], max_trials=maxTrial, directory=autoKerasPath)

    classifier = ak.ImageClassifier(max_trials=maxTrial, directory=autoKerasPath)

    return classifier
