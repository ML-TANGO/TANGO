import express from "express"
import asyncHandler from "express-async-handler"
import multer from "multer"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import fse from "fs-extra"
import rimraf from "rimraf"
import getSize from "get-folder-size"

import DH from "../../lib/DatabaseHandler"
import C from "../../lib/CommonConstants"
import CL from "../../lib/ConfigurationLoader"
import CF from "../../lib/CommonFunction"
import { resolve } from "app-root-path"
import { reject } from "bluebird"
var logger = require("../../lib/Logger")(__filename)

const router = express.Router()
const config = CL.getConfig()
const CRN_USR = "testUser@test.co.kr"

router.post("/stoptrain", async (req, res, next) => {
  let status = 1
  let message = ""
  try {
    let param = {}
    param.AI_CD = req.body.AI_CD
    let path = await _exeQuery(C.Q.AIPRJ, param, "getAiPath")
    path = path[0].PATH

    let result = await _exeQuery(C.Q.TABAIPRJ, param, "getAiProjectInfo")
    result = result[0]

    if (result.AI_PID !== null) {
      let data = await CF.runProcess("kill", [result.AI_PID]).catch((err) => {
        logger.error(
          `!!! [${param.AI_CD}][PID: ${result.AI_PID}] kill process Fail  \n${err.stack}`
        )
        status = 0
        message = err.stack
      })

      param.AI_STS = "DONE"
      param.AI_PID = null
      param.LAST_MSG = "User Stop Model"
      if (path !== null && path !== "") {
        getSize(path, (err, size) => {
          param.AI_SIZE = size
        })
      }
      await _exeQuery(C.Q.TABAIPRJ, param, "updateAiStatus")

      param.MDL_STS = "FAIL"
      param.MDL_IDX = null
      await _exeQuery(C.Q.TABAIPRJ, param, "updateModelStatus")
    }
  } catch (error) {
    status = 0
    message = err.stack
  }
  res.json({ status: status, message: message })
})

router.post("/getmodelsummary", async (req, res, next) => {
  const param = req.body
  let data = await _exeQuery(C.Q.TABAIPRJ, param, "getModelSummaryValue")
  res.json(data)
})

router.post("/getmodelresult", async (req, res, next) => {
  const ai_cd = req.body.AI_CD
  const mdlIdxList = req.body.MDL_IDX_LIST
  const aiPath = path.join(config.aiPath, ai_cd)
  const graphType = req.body.GRAPH_TYPE

  let results = []
  mdlIdxList.map((ele) => {
    if (graphType === undefined) {
      try {
        let aiMdlPath = path.join(aiPath, String(ele.MDL_IDX))
        fs.readdirSync(aiMdlPath).map((fileName) => {
          if (fileName.includes(".chart")) {
            const chartFile = path.join(aiMdlPath, fileName)
            results.push(JSON.parse(fs.readFileSync(chartFile)))
          }
        })
      } catch (error) {}
    } else {
      try {
        const filePath = path.join(
          path.join(aiPath, String(ele.MDL_IDX)),
          `${graphType}.chart`
        )
        results.push(JSON.parse(fs.readFileSync(filePath)))
      } catch (error) {}
    }
  })
  res.json(results)
})

router.post("/getmodellist", async (req, res, next) => {
  let param = {}
  let list = {}
  let db_list = await _exeQuery(C.Q.TABAIPRJ, param, "getLegacyModelOptions")
  // console.log(list)

  db_list.map((ele, i) => {
    if (list[ele.MDL_KIND] === undefined) list[ele.MDL_KIND] = {}
    if (list[ele.MDL_KIND][ele.OBJECT_TYPE] === undefined)
      list[ele.MDL_KIND][ele.OBJECT_TYPE] = []
    let paramBody = JSON.parse(ele.PARAM)
    paramBody.MODEL_NAME = ele.MDL_NM
    list[ele.MDL_KIND][ele.OBJECT_TYPE].push(paramBody)
  })

  res.json(list)
})

router.post("/starttrain", async (req, res, next) => {
  let result = await _getAiData(req.body.AI_CD)
  result.datasetList = result.datasetList[0]

  let param = {}
  param.AI_CD = req.body.AI_CD
  param.DATASET_CD = result.datasetList.DATASET_CD

  let inputData = {}
  let serverParam = {}
  let modelInfo = {}

  const fe_type = String(param.DATASET_CD).substr(0, 1)

  try {
    //인풋데이터 설정/////
    let trainPath = []
    let testPath = null
    trainPath = await _exeQuery(C.Q.TABAIPRJ, param, "getFileList")
    if (result.datasetList.TEST_DATASET_CD !== null) {
      testPath = await _exeQuery(
        C.Q.TABAIPRJ,
        { DATASET_CD: result.datasetList.TEST_DATASET_CD },
        "getFileList"
      )
    }

    result.datasetList.columnList = result.datasetList.columnList.filter(
      (ele) => ele.checked === 1
    )

    let targetNm = ""
    result.datasetList.columnList.map((ele) => {
      if (ele.IS_CLASS === 1) targetNm = ele.COLUMN_NM
    })

    if (targetNm === "") throw new Error("Target column not exist")

    let inputData = {
      TRAIN_PATH: trainPath,
      DELIMITER: ",", //미정
      SPLIT_YN: result.datasetList.SPLIT_YN,
      DATASET_SPLIT: result.datasetList.DATASET_SPLIT,
      TEST_DATASET_CD: result.datasetList.TEST_DATASET_CD,
      TEST_PATH: testPath,
      LABEL_COLUMN_NAME: targetNm,
      MAPING_INFO: result.datasetList.columnList,
      FE_TYPE: fe_type
    }

    if (result.datasetList.UPLOAD_TYPE === "DB") {
      inputData.DB_INFO = await _exeQuery(
        C.Q.TABDATASET,
        { DATASET_CD: param.DATASET_CD, KEY: "WWED" },
        "getDBInfo"
      )
      inputData.DB_INFO = inputData.DB_INFO[0]
      inputData.DB_INFO.PASSWORD = String(inputData.DB_INFO.PASSWORD)
    }

    logger.info(`Set Input Data [${req.body.AI_CD}]`)

    //서버 전송 파라미터 설정/////
    serverParam = {
      AI_CD: param.AI_CD,
      AI_PATH: path.join(config.aiPath, param.AI_CD),
      SRV_IP: "127.0.0.1", /// 서버 변경
      SRV_PORT: 10236,
      TRAIN_RESULT_URL: "/tab/binary/trainResultLog",
      TRAIN_STATE_URL: "/tab/binary/binaryStatusLog",
      TRAINING_INFO_URL: "/tab/binary/trainInfoLog"
    }

    //각 모델 정보 설정/////
    modelInfo = []

    let getModelInfo = result.modelList.map((ele) => {
      return new Promise(async (resolve, reject) => {
        ele.MDL_PATH = await _exeQuery(C.Q.TABAIPRJ, ele, "getLegacyModelInfo")
        ele.MDL_PATH = ele.MDL_PATH[0].MDL_PATH
        resolve(ele)
      })
    })

    await Promise.all(getModelInfo).then((data) => {
      modelInfo.push(data)
    })

    const trainARGS = {
      INPUT_DATA: inputData,
      SERVER_PARAM: serverParam,
      MODEL_INFO: modelInfo[0]
    }

    logger.info(`[${req.body.AI_CD}] Set Server Param / Model Info Data `)
    // console.log(JSON.stringify(trainARGS, null, 1))
    await CF.createProcess("python", [
      C.BIN.runTrain,
      `${JSON.stringify(trainARGS)}`
    ])
    res.json({ status: 1 })
  } catch (error) {
    logger.error(`[${req.body.AI_CD}] Train Start Fail \n${error.stack}`)
    res.json({ status: 0 })
  }
})

const _getAiData = async (AI_CD) => {
  let param = {}
  param.AI_CD = AI_CD

  /////////// AI 기본정보 조회////////////////
  let result = await _exeQuery(C.Q.TABAIPRJ, param, "getAiProjectInfo")
  result = result[0]

  /////////// 트레인 데이터셋 조회////////////////
  let datasetList = await _exeQuery(C.Q.TABAIPRJ, param, "getFeatureSet")
  datasetList[0].AI_CD = param.AI_CD

  /////////// 트레인 데이터셋의 컬럼리스트, 선택여부 조회////////////////
  let columnList = await _exeQuery(
    C.Q.TABAIPRJ,
    datasetList[0],
    "getTrainFeatures"
  )
  datasetList[0].columnList = columnList

  /////////// 모델 파라미터설정 저장  ////////////////
  let modelListDB = await _exeQuery(C.Q.TABAIPRJ, param, "getTrainModel")
  let modelList = []
  modelListDB.map((ele) => {
    modelList.push(JSON.parse(ele.PARAM))
  })

  result.datasetList = datasetList
  result.modelList = modelList
  return result
}

router.post("/getaiprojectdata", async (req, res, next) => {
  let result = await _getAiData(req.body.AI_CD)
  res.json(result)
})

router.post("/updateaiproject", async (req, res, next) => {
  let param = {}
  param = req.body
  param.DATA_TYPE = "T"
  param.YEAR = moment().format("YYYY")

  // let prjNumber = param.datasetList[0].AI_CD
  // param.AI_CD = prjNumber

  ///기존 데이터 삭제
  await _exeQuery(C.Q.TABAIPRJ, param, "updateAiStatus")
  await _exeQuery(C.Q.TABAIPRJ, param, "removeTrainSet")
  await _exeQuery(C.Q.TABAIPRJ, param, "removeTrainFeatures")
  await _exeQuery(C.Q.TABAIPRJ, param, "removeTrainModelInfo")
  await _exeQuery(C.Q.TABAIPRJ, param, "removeModelSummary")

  // rimraf.sync(path.join(config.aiPath, param.AI_CD))
  logger.debug(`[${param.AI_CD}] remove Old Data`)

  //////데이터셋 삽입//////////
  let datasetParam = param.datasetList[0]
  datasetParam.AI_CD = param.AI_CD
  datasetParam.TESTDATASET =
    datasetParam.TEST_DATASET_CD === undefined
      ? null
      : datasetParam.TEST_DATASET_CD
  await _exeQuery(C.Q.TABAIPRJ, datasetParam, "setTrainSet")

  datasetParam.columnList.map((ele) => {
    if (ele.checked === 1 || ele.checked === true) ele.checked = true
    else ele.checked = false
  })
  //////트레인에 사용될 컬럼 저장//////////
  await _exeQuery(C.Q.TABAIPRJ, datasetParam, "setTrainFeatures")

  //////트레인 학습 설정 저장//////////
  let trainParam = {}
  trainParam.AI_CD = param.AI_CD
  trainParam.DATA = []
  let idx = 0
  param.modelList.map((ele) => {
    trainParam.DATA.push({
      NETWORK_NAME: ele.MODEL_NAME,
      MDL_IDX: idx,
      MDL_ALIAS: ele.MDL_ALIAS === undefined ? null : ele.MDL_ALIAS,
      PARAM: JSON.stringify(ele)
    })
    idx++
  })
  await _exeQuery(C.Q.TABAIPRJ, trainParam, "setTrainModelInfo")
  res.json({ status: 1 })
})

router.post("/createaiproject", async (req, res, next) => {
  let param = {}
  param = req.body
  param.DATA_TYPE = "T"
  param.YEAR = moment().format("YYYY")

  console.log(param)

  ///////////채번/////////////////
  await _exeQuery(C.Q.AIPRJ, param, "setNewPrjNumber")
  let prjNumber = await _exeQuery(C.Q.AIPRJ, param, "getNewPrjNumber")
  prjNumber = prjNumber[0].PRJ_NUMBER

  ////////AI 생성//////////////
  param.AI_CD = prjNumber
  param.AI_TYPE = param.OBJECT_TYPE + param.DATA_TYPE
  param.CRN_USR = req.body.USER_ID
  param.PATH = path.join(config.aiPath, prjNumber)
  await _exeQuery(C.Q.AIPRJ, param, "setAiPrj")

  !fs.existsSync(param.PATH) && fs.mkdirSync(param.PATH)

  //////데이터셋 삽입//////////
  let datasetParam = param.datasetList[0]
  datasetParam.AI_CD = param.AI_CD
  datasetParam.TESTDATASET =
    datasetParam.TEST_DATASET_CD === undefined
      ? null
      : datasetParam.TEST_DATASET_CD
  await _exeQuery(C.Q.TABAIPRJ, datasetParam, "setTrainSet")

  //////트레인에 사용될 컬럼 저장//////////
  await _exeQuery(C.Q.TABAIPRJ, datasetParam, "setTrainFeatures")

  //////트레인 학습 설정 저장//////////
  let trainParam = {}
  trainParam.AI_CD = param.AI_CD
  trainParam.DATA = []
  let idx = 0
  param.modelList.map((ele) => {
    trainParam.DATA.push({
      NETWORK_NAME: ele.MODEL_NAME,
      MDL_IDX: idx,
      MDL_ALIAS: ele.MDL_ALIAS === undefined ? null : ele.MDL_ALIAS,
      PARAM: JSON.stringify(ele)
    })
    idx++
  })
  await _exeQuery(C.Q.TABAIPRJ, trainParam, "setTrainModelInfo")
  res.json({ status: 1 })
})

const _exeQuery = async (source, param, queryId) => {
  return new Promise((resolve, reject) => {
    let option = {}
    option.source = source
    option.param = param
    option.queryId = queryId
    DH.executeQuery(option)
      .then((data) => {
        resolve(data)
      })
      .catch((err) => {
        reject(err)
      })
  })
}

module.exports = router
