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
var logger = require("../../lib/Logger")(__filename)

const router = express.Router()
const config = CL.getConfig()
const CRN_USR = "testUser@test.co.kr"

const _sendTrainSocData = (type, AI_CD, MSG, AI_STS) => {
  let sendData = {}
  sendData.TYPE = type
  sendData.AI_CD = AI_CD
  sendData.MSG = MSG
  sendData.AI_STS = AI_STS
  CF.sendSoc(sendData, "TRAIN")
}

router.post("/trainResultLog", async (req, res, next) => {
  // console.log(req.body)
  //////////각 모델별 결과정보
  const AI_CD = req.body.GRAPH_INFO[0].AI_CD
  const mdlIdx = String(req.body.GRAPH_INFO[0].MDL_IDX)
  const mdlName = req.body.GRAPH_INFO[0].MODEL_NAME
  const body = req.body.GRAPH_INFO
  const aiPath = path.join(path.join(config.aiPath, AI_CD), mdlIdx)
  try {
    body.map((ele) => {
      const filePath = path.join(aiPath, `${ele.GRAPH_TYPE}.chart`)
      fs.writeFileSync(filePath, JSON.stringify(ele), {
        encoding: "utf8",
        flag: "w"
      })
    })

    let param = req.body.SCORE_INFO
    param.AI_CD = AI_CD
    param.EPOCH = 0
    param.OBJECT_TYPE = AI_CD.substring(0, 1)
    param.DATA_TYPE = AI_CD.substring(1, 2)
    param.CRN_USR = CRN_USR
    param.MDL_IDX = mdlIdx
    param.IS_PREDICT = true

    await _exeQuery(C.Q.TABAIPRJ, param, "setTrainPrediceResult")

    logger.info(`[${AI_CD}][${mdlIdx}] - [${mdlName}] Result Save Success`)
  } catch (error) {
    logger.error(
      `[${AI_CD}][${mdlIdx}] - [${mdlName}] Result Save Fail\n${error.stack}`
    )
  }

  res.json({ status: 1 })
})

router.post("/binaryStatusLog", async (req, res, next) => {
  // console.log(req.body)
  //////////각 모델별 생존 정보
  try {
    const mdlStatus = req.body.STATUS
    const body = req.body
    let param = {}
    param.MDL_IDX = body.MDL_IDX
    param.AI_CD = body.AI_CD
    if (mdlStatus) {
      //처음 생존 신호
      // console.log("====================")
      // console.log(req.body)
      // console.log("====================")

      param.MDL_STS = "LEARN"
      await _exeQuery(C.Q.TABAIPRJ, param, "updateModelStatus")
      _sendTrainSocData("STATE_CHANGE_MDL", body.AI_CD, body, "LEARN")
    } else {
      //종료 생존 신호
      if (body.MSG === "train done") {
        param.MDL_STS = "DONE"
        logger.info(
          `[${body.AI_CD}][${body.MDL_IDX}] - [${body.MODEL_NAME}] Trainning END`
        )
      } else {
        param.MDL_STS = "FAIL"
        logger.error(
          `[${body.AI_CD}][${body.MDL_IDX}] - [${body.MODEL_NAME}] Trainning FAIL\nERR_MSG: ${body.MSG}`
        )
      }
      param.LAST_MSG = body.MSG
      await _exeQuery(C.Q.TABAIPRJ, param, "updateModelStatus").catch((err) => {
        console.log("===============")
        console.log(req.body)
        console.log("===============")
      })
      _sendTrainSocData("STATE_CHANGE_MDL", body.AI_CD, body, param.MDL_STS)
    }
  } catch (error) {
    logger.error(
      `[${req.body.AI_CD}] Train FAIL (not Train Manager) \n${error.stack}`
    )
  }
  res.json({ status: 1 })
})

router.post("/trainInfoLog", async (req, res, next) => {
  //////////모델의 각 이포크별 정보
  let param = req.body
  if (param.EPOCH === 1) {
    await _exeQuery(C.Q.TABAIPRJ, param, "initEpoch")
    logger.info(`[${param.AI_CD}][${param.MDL_IDX}] - EPOCH Init`)
  }

  param.CRN_USR = CRN_USR
  await _exeQuery(C.Q.TABAIPRJ, param, "setTrainEpoch")
  _sendTrainSocData("INFO_UPDATE", param.AI_CD, param, "LEARN")
  // if (param.EPOCH % 10 === 0)
  logger.info(`    [${param.AI_CD}][${param.MDL_IDX}] - EPOCH (${param.EPOCH})`)
  res.json({ status: 1 })
})

router.post("/trainmanagerinfo", async (req, res, next) => {
  //////////트레인 메니저에 대한 생존 API
  try {
    const managerStatus = req.body.STATUS
    const body = req.body
    let param = {}
    param.AI_CD = body.AI_CD
    let path = await _exeQuery(C.Q.AIPRJ, param, "getAiPath")
    path = path[0].PATH
    if (path !== null && path !== "") {
      getSize(path, (err, size) => {
        param.AI_SIZE = size
      })
    }
    param.AI_PID = body.PID
    if (managerStatus) {
      //트레인 메니저 스타트
      param.AI_STS = "LEARN"
      param.TRAIN_SRT_DTM = "now()"
      param.LAST_MSG = body.MSG
      await _exeQuery(C.Q.TABAIPRJ, param, "updateAiStatus")
      logger.info(`[${body.AI_CD}][TRAIN MANAGER] Trainning Start`)
    } else {
      param.AI_PID = null
      param.TRAIN_END_DTM = "now()"
      param.LAST_MSG = body.MSG
      if (body.PID === null) {
        //트레인 메니저 비정상 종료. 에러메시지 포함
        param.AI_STS = "FAIL"
        await _exeQuery(C.Q.TABAIPRJ, param, "updateAiStatus")
        logger.error(
          `[${body.AI_CD}][TRAIN MANAGER] Trainning FAIL \n${body.MSG}`
        )
      } else {
        //트레인 메니저 정상 종료
        param.AI_STS = "DONE"
        if (path !== null && path !== "") {
          getSize(path, (err, size) => {
            param.AI_SIZE = size
          })
        }
        await _exeQuery(C.Q.TABAIPRJ, param, "updateAiStatus")
        logger.info(`[${body.AI_CD}][TRAIN MANAGER] Trainning END`)
      }
    }
    _sendTrainSocData("STATE_CHANGE", param.AI_CD, param, param.AI_STS)
  } catch (error) {
    logger.error(`[${req.body.AI_CD}] Train Manager FAIL \n${error.stack}`)
  }
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
