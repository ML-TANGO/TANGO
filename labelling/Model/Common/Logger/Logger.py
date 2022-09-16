# -*- coding:utf-8 -*-
'''
Binary log 수집 스크립트
1. Log 파일은 일자 별로 /Logs/Model/*.log 파일에 쌓이도록 함.
2. Log는 [INFO], [ERROR] 두가지 임.
3. ERROR 발생 시, 어떤 에러가 어떤 스크립트에서 발생했는지 표기
'''
import os
import sys
import time
import logging
import logging.handlers

# Model 폴더 찾기 위한 방법
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir, "../")))

# Model Path 등록
sys.path.append(basePath)
from Common.Utils.Utils import getConfig, getLogLevel
srvIp, srvPort, logPath, tempPath, datasetPath, aiPath = getConfig()

logLevel = getLogLevel()


def logger(loggerName):
    global logLevel
    # Create Logger
    global logPath

    logger = logging.getLogger(loggerName)

    if not os.path.isdir(logPath):
        os.makedirs(logPath, exist_ok=True)

    tmp = time.strftime('%Y-%m-%d', time.localtime(time.time()))

    logFileName = 'Model-{}.log'.format(tmp)
    if len(logger.handlers) > 0:
        # Logger already exists
        return logger

    # stream logger
    # streamHandler = logging.StreamHandler()

    # file logger
    if not os.path.isdir(logPath):
        os.mkdir(logPath)

    # log Path
    fileName = os.path.join(logPath, logFileName)
    fileHandler = logging.FileHandler(fileName, mode="a")

    # logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(logging.INFO) if logLevel == "INFO" else logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s [Filename : %(filename)s]', "%Y-%m-%d %H:%M:%S")
    # formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s ', "%Y-%m-%d %H:%M:%S")

    # Create Handlers
    # streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    return logger
