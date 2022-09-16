import express from "express"
import asyncHandler from "express-async-handler"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import { parse } from "fast-csv"
import gm from "gm"
import ncp from "ncp"
import ThumbnailGenerator from "video-thumbnail-generator"
var crypto = require("crypto-js")
import si from "systeminformation"
import unzip from "node-unzip-2"
import tar from "tar"
import rimraf from "rimraf"

import DH from "../lib/DatabaseHandler"
import C from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"
import EXDB from "../lib/ExtDataBaseHandler"
import BL from "../lib/BuildLoader"
import { checkServerIdentity } from "tls"

import passport from "passport"
import {
  register,
  changepw,
  login,
  refreshToken,
  logout
} from "../lib/Authentication"

var logger = require("../lib/Logger")(__filename)

const router = express.Router()
const config = CL.getConfig()
const CRN_USR = "testUser@test.co.kr"
const spawn = require("child_process").spawn

router.post("/gettabledata", async (req, res, next) => {
  try {
    let param = req.body
    const start = req.body.START
    const end = req.body.END

    let searchData = []

    let dataset = await CF.exeQuery(C.Q.DATASET, param, "getDataSet")
    dataset = dataset[0]

    if (dataset.UPLOAD_TYPE === "DB") {
      param.KEY = "WWED"
      let dbInfo = await CF.exeQuery(C.Q.TABDATASET, param, "getDBInfo")
      dbInfo = dbInfo[0]
      dbInfo.PASSWORD = String(dbInfo.PASSWORD)
      dbInfo.LIMIT = req.body.END

      logger.info(
        `Starting get DB Table Data in [${dbInfo.ADDRESS}:${dbInfo.DBNAME}] (${dbInfo.CLIENT})`
      )
      let result = await EXDB.extDBConnection(dbInfo)
      searchData = result.DATA.slice(req.body.START, result.DATA.length)
      logger.info(
        `Success get DB Table Data (${req.body.START} - ${result.DATA.length} : ${searchData.length}row)`
      )
    } else {
      let fileList = await CF.exeQuery(C.Q.DATASET, dataset, "getFileList")
      logger.info(
        `Starting get File Table Data in (${JSON.stringify(fileList, null, 1)})`
      )
      searchData = await _getTableDataInFile([], fileList, 0, 0, start, end)
      logger.info(
        `Success get File Table Data (${start} - ${end} : ${searchData.length}row)`
      )
    }

    res.json(searchData)
  } catch (error) {
    console.log(error)
    res.json({ status: 0, msg: error })
    logger.error(error)
  }
})

const _getTableDataInFile = async (
  searchData,
  fileList,
  fileIdx,
  rowIdx,
  start,
  end
) => {
  if (fileList[fileIdx] !== undefined) {
    const stream = fs
      .createReadStream(fileList[fileIdx].FILE_PATH)
      .pipe(parse({ headers: true }))

    return new Promise((resolve, reject) => {
      stream
        .on("error", (error) => logger.error(error))
        .on("data", (row) => {
          if (rowIdx >= start && rowIdx <= end) {
            searchData.push(row)
            // console.log(row)
          }
          rowIdx++
        })
        .on("end", (rowCount) => {
          if (rowIdx < end) {
            logger.debug("keep going Search")
            fileIdx++
            _getTableDataInFile(
              searchData,
              fileList,
              fileIdx,
              rowIdx,
              start,
              end
            )
          }
          resolve(searchData)
        })
    })
  }
}

router.post("/gettableresult", async (req, res, next) => {
  let param = req.body
  const start = req.body.START
  const end = req.body.END
  let argData = {}

  let labelColumn = await CF.exeQuery(C.Q.TABDATASET, param, "getLabelColumn")
  labelColumn = labelColumn[0].COLUMN_NM

  let dataset = await CF.exeQuery(C.Q.DATASET, param, "getDataSet")
  dataset = dataset[0]

  argData.INPUT_DATA = {
    START: start,
    END: end,
    LABEL_COLUMN_NAME: labelColumn,
    UPLOAD_TYPE: dataset.UPLOAD_TYPE
  }
  argData.INPUT_DATA.TEST_PATH = []
  argData.INPUT_DATA.DB_INFO = {}

  if (dataset.UPLOAD_TYPE === "DB") {
    param.KEY = "WWED"
    let dbInfo = await CF.exeQuery(C.Q.TABDATASET, param, "getDBInfo")
    dbInfo = dbInfo[0]
    dbInfo.PASSWORD = String(dbInfo.PASSWORD)
    dbInfo.LIMIT = req.body.END
    logger.info(
      `Success get DB Table Data (${req.body.START} - ${dbInfo.length} row)`
    )
    argData.INPUT_DATA.DB_INFO = dbInfo
  } else {
    argData.INPUT_DATA.TEST_PATH = []
    let fileList = await CF.exeQuery(C.Q.DATASET, dataset, "getFileList")
    fileList.map((ele) => {
      argData.INPUT_DATA.TEST_PATH.push(ele.FILE_PATH)
    })
  }

  let legacyInfo = await CF.exeQuery(C.Q.TABAIPRJ, param, "getTabWeightPath")
  legacyInfo = legacyInfo[0]
  const mdlPath = legacyInfo.MDL_PATH
  const objType = legacyInfo.OBJECT_TYPE
  const dataType = legacyInfo.DATA_TYPE

  let weightPath = path.join(config.aiPath, param.AI_CD)
  weightPath = path.join(weightPath, String(param.MDL_IDX))

  argData.MODEL_INFO = {
    MODEL_PATH: mdlPath,
    WEIGHT_PATH: weightPath,
    DATA_TYPE: dataType,
    OBJECT_TYPE: objType
  }

  try {
    let Result = {}
    let data = await CF.runProcess("python", [
      C.BIN.runEvaluation,
      JSON.stringify(argData)
    ])
    Result = JSON.parse(data.stdout)
    let cnt = 0
    let avg = 0

    // if(Result[0].STATUS === 0) throw new Error(Result[0].MSG)
    if (Result.RESULT !== undefined)
      Result.RESULT.map((ele) => {
        if (ele.ACCURACY !== undefined) {
          avg += ele.ACCURACY
          cnt++
        }
      })
    Result.AVG_ACC = (avg / cnt).toFixed(3)
    res.json(Result)
  } catch (err) {
    logger.error(`predict Fail \n${err}`)
    res.json({ status: 0, msg: err })
  }
})

module.exports = router
