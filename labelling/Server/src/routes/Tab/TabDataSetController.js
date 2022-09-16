import express from "express"
import asyncHandler from "express-async-handler"
import multer from "multer"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import fse from "fs-extra"
import rimraf from "rimraf"
import { parse } from "fast-csv"
import typeCheck from "type-check"

import DH from "../../lib/DatabaseHandler"
import C from "../../lib/CommonConstants"
import CL from "../../lib/ConfigurationLoader"
import CF from "../../lib/CommonFunction"
import EXDB from "../../lib/ExtDataBaseHandler"

var logger = require("../../lib/Logger")(__filename)

const router = express.Router()
const config = CL.getConfig()
const CRN_USR = "testUser@test.co.kr"

let storage = multer.diskStorage({
  destination: (req, file, cb) => {
    let fileNames = file.originalname.replace(/_@_/g, "/")
    // // file.originalname = fileNames[1]
    !fs.existsSync(path.join(config.tempPath, path.dirname(fileNames))) &&
      fs.mkdirSync(path.join(config.tempPath, path.dirname(fileNames)), {
        recursive: true
      })

    cb(null, path.join(config.tempPath, path.dirname(fileNames)))
  },
  filename: (req, files, cb) => {
    let savedFile = files.originalname.replace(/ /g, "")
    savedFile = path.basename(savedFile.replace(/_@_/g, "/"))
    cb(null, savedFile) // cb 콜백함수를 통해 전송된 파일 이름 설정
  }
})

const upload = multer({ storage: storage })

router.post(
  "/upload",
  upload.array("file"),
  asyncHandler(async (req, res, next) => {
    // min gi
    let fileList = JSON.parse(req.body.dir)
    const promiseList = fileList.map((el) => {
      return new Promise((resolve, reject) => {
        let filePath = path.join(config.tempPath, req.body.uuid)
        filePath = path.join(filePath, el)
        CF.runProcess("python", [C.BIN.getHeader, filePath])
          .then((result) => {
            let column = JSON.parse(result.stdout)
            resolve({
              path: el,
              fileName: path.basename(el),
              status: 1,
              columns: column.COLUMNS
            })
          })
          .catch((err) => {
            reject(err)
            logger.error(`getHeader Fail \n${err.stderr}`)
          })
      })
    })

    Promise.all(promiseList).then((result) => {
      res.json(result)
    })
  })
)

router.post("/getfiledata", async (req, res, next) => {
  console.log("go")
  // var io = require("socket.io").listen(80)
  // var ss = require("socket.io-stream")
  // var path = require("path")

  // io.of("/user").on("connection", function (socket) {
  //   ss(socket).on("profile-image", function (stream, data) {
  //     console.log("data data " + data)
  //     var filename = path.basename(data.name)
  //     stream.pipe(
  //       fs.createWriteStream("/Users/dmshin/Downloads/kddcup99_csv.csv")
  //     )
  //   })
  // })
  // res.json({ status: 1 })
})

router.post("/getfeatures", async (req, res, next) => {
  let param = req.body
  let features = await _exeQuery(C.Q.TABDATASET, param, "getFeatures")
  res.json(features)
})

router.post("/getdbinfo", async (req, res, next) => {
  try {
    let param = req.body
    param.KEY = "WWED"
    let dbInfo = await _exeQuery(C.Q.TABDATASET, param, "getDBInfo")
    dbInfo = dbInfo[0]
    dbInfo.PASSWORD = String(dbInfo.PASSWORD)
    res.json({ table: dbInfo })
  } catch (error) {
    res.json({ status: 0, msg: error.stack })
  }
})

router.post("/setUpdateDataset", async (req, res, next) => {
  let param = {}
  let rollback = {}
  rollback.orgData = {}
  rollback.newData = {}

  param = req.body

  const datasetDir = path.join(config.datasetPath, req.body.DATASET_CD)
  const tempDir = path.join(config.tempPath, req.body.uuid)
  req.body.DATASET_DIR = datasetDir
  // await DH.executeQuery(option)
  await _exeQuery(C.Q.DATASET, param, "setUpdateDataset")
  if (req.body.remove !== undefined && req.body.remove.length > 0) {
    let removeFiles = req.body.remove.map((file) => {
      return `'${path.parse(file).name}'`
    })
    removeFiles = removeFiles.join(",")

    param.DATASET_CD = req.body.DATASET_CD
    param.FILE_NAMES = removeFiles
    await _exeQuery(C.Q.DATASET, param, "removeDataElementsByName")

    req.body.remove.map((removeFile) => {
      let fileName = path.join(datasetDir, removeFile)
      fse.removeSync(fileName)
    })
  }

  let fileList = req.body.fileList
  fileList.map((ele) => {
    ele.path = ele.path.replace(/ /g, "")
    // ele.FILE_TYPE = path.parse(path.basename(ele.path)).ext
    let tgtPath = path.join(datasetDir, ele.path)
    if (!fs.existsSync(path.dirname(tgtPath)))
      fs.mkdirSync(path.dirname(tgtPath), { recursive: true })
    fs.copyFileSync(path.join(tempDir, ele.path), tgtPath)
  })
  rollback.orgData.files = fileList
  rollback.orgData.tempPath = tempDir
  rollback.newData.dataPath = req.body.DATASET_CD

  let dataCd = await _exeQuery(C.Q.DATASET, param, "getMaxDataCd")
  dataCd = Number(dataCd[0].DATA_CD) + 1

  param = {}
  param.DATASET_CD = req.body.DATASET_CD
  param.DATASET_STS = "CREATE"
  await _exeQuery(C.Q.DATASET, param, "updateDataSetStatus")

  req.body.files = fileList
  // req.body.AUTO_TYPE = "N"
  _createDataSets(req.body, dataCd)

  res.json({ status: 1 })
})

router.post("/setdupdataset", async (req, res, next) => {
  //데이터셋 채번

  const tempDir = path.join(config.datasetPath, req.body.ORG_DATASET_CD)
  let param = req.body
  param.YEAR = moment().format("YY")
  //신규 데이터셋 코드 생성
  await _exeQuery(C.Q.DATASET, param, "setNewDataSetNumber")
  //신규 데이터셋 조회
  let datasetNo = await _exeQuery(C.Q.DATASET, param, "getNewDataSetNumber")

  datasetNo = datasetNo[0].DATASET_NUMBER
  req.body.DATASET_CD = datasetNo

  const datasetDir = path.join(config.datasetPath, datasetNo)
  req.body.DATASET_DIR = datasetDir

  //파일 이동
  try {
    !fs.existsSync(datasetDir) && fs.mkdirSync(datasetDir)

    // 신규 Dataset 폴더로 복사
    if (req.body.UPLOAD_TYPE !== "DB" && req.body.UPLOAD_TYPE !== undefined)
      fse.copySync(tempDir, datasetDir)

    let resultPath = path.join(datasetDir, "result")
    !fs.existsSync(resultPath) && fs.mkdirSync(resultPath)
  } catch (error) {
    logger.error(error.message)
    res.json({ status: 0, err: error.message, msg: "폴더생성실패" })
    return
  }
  logger.debug("ORG file copy Done")

  //DB 정보 저장

  let fileList = req.body.files
  fileList.map((ele) => {
    ele.path = ele.path.replace(/ /g, "")
    ele.path = ele.path.replace(tempDir, "")
  })

  param.DATASET_CD = datasetNo
  param.CRN_USR = param.USER_ID
  param.THUM_NAIL_CD = "T0000000"
  param.CATEGORY1 = "USER"
  await _exeQuery(C.Q.DATASET, param, "setNewDataSet")
  param = {}
  param.DATASET_CD = datasetNo
  param.DATASET_STS = "CREATE"
  await _exeQuery(C.Q.DATASET, param, "updateDataSetStatus")

  res.json({ status: 1 })

  _createDataSets(req.body, 0)
  //파일리스트 업데이트, 피처리스트 업데이트
})

router.post("/createdataset", async (req, res, next) => {
  //데이터셋 채번
  let rollback = {}
  rollback.orgData = {}
  rollback.newData = {}

  const tempDir = path.join(config.tempPath, req.body.uuid)
  let param = req.body
  param.YEAR = moment().format("YY")

  //신규 데이터셋 코드 생성
  await _exeQuery(C.Q.DATASET, param, "setNewDataSetNumber")
  //신규 데이터셋 조회
  let datasetNo = await _exeQuery(C.Q.DATASET, param, "getNewDataSetNumber")

  datasetNo = datasetNo[0].DATASET_NUMBER
  req.body.DATASET_CD = datasetNo

  const datasetDir = path.join(config.datasetPath, datasetNo)
  req.body.DATASET_DIR = datasetDir

  //파일 이동

  try {
    !fs.existsSync(datasetDir) && fs.mkdirSync(datasetDir)

    if (req.body.UPLOAD_TYPE !== "DB" && req.body.UPLOAD_TYPE !== undefined)
      fs.renameSync(tempDir, datasetDir)

    let resultPath = path.join(datasetDir, "result")
    !fs.existsSync(resultPath) && fs.mkdirSync(resultPath)
  } catch (error) {
    logger.error(error.message)
    res.json({ status: 0, err: error.message, msg: "폴더생성실패" })
    return
  }
  logger.debug("Temp file copy Done")

  //DB 정보 저장

  let fileList = req.body.files
  fileList.map((ele) => {
    ele.path = ele.path.replace(/ /g, "")
  })

  rollback.orgData.files = fileList
  rollback.orgData.tempPath = tempDir
  rollback.newData.dataPath = datasetDir

  param.DATASET_CD = datasetNo
  param.CRN_USR = param.USER_ID
  param.THUM_NAIL_CD = "T0000000"
  param.CATEGORY1 = "USER"
  await _exeQuery(C.Q.DATASET, param, "setNewDataSet")
  param = {}
  param.DATASET_CD = datasetNo
  param.DATASET_STS = "CREATE"
  await _exeQuery(C.Q.DATASET, param, "updateDataSetStatus")

  res.json({ status: 1 })

  _createDataSets(req.body, 0)
  //파일리스트 업데이트, 피처리스트 업데이트
})

const _createDataSets = async (data, dataCd) => {
  let fileList = data.files
  const datasetDir = data.DATASET_DIR
  const DATASET_CD = data.DATASET_CD

  let param = {}

  try {
    //피처셋 업데이트
    param.DATA = []
    param.DATASET_CD = DATASET_CD
    data.COLUMNS.map((ele, idx) => {
      param.DATA.push({
        DATASET_CD: DATASET_CD,
        COLUMN_NM: ele.COLUMN_NM,
        COLUMN_ALIAS: null,
        DEFAULT_VALUE: ele.DEFAULT_VALUE === "null" ? null : ele.DEFAULT_VALUE,
        IS_CLASS: ele.COLUMN_NM === data.TARGET ? 1 : 0,
        COLUMN_IDX: idx
      })
    })

    _exeQuery(C.Q.TABDATASET, param, "removeFeatures")
    _exeQuery(C.Q.TABDATASET, param, "setFeatureInfo")

    //파일리스트 업데이트
    param.DATA = []
    let filePaths = []
    fileList.map(async (ele, idx) => {
      let tempEle = {
        DATASET_CD: DATASET_CD,
        DATA_CD: String(dataCd + idx).padStart(8, 0),
        DATA_STATUS: "ORG",
        FILE_NAME: path.parse(ele.path).name,
        FILE_EXT: path.parse(ele.path).ext,
        FILE_TYPE: data.DATA_TYPE,
        FILE_PATH: path.join(datasetDir, ele.path),
        FILE_RELPATH: ele.path,
        FILE_SIZE: ele.size,
        FPS: ele.FPS === undefined ? 0 : ele.FPS,
        TAG_CD: 0
      }
      param.DATA.push(tempEle)
      filePaths.push(path.join(datasetDir, ele.path))
    })

    if (data.UPLOAD_TYPE === "DB") {
      data.table.DATASET_CD = DATASET_CD
      data.table.KEY = "WWED"
      data.table.DB_SEQ = 0

      await _exeQuery(C.Q.TABDATASET, data.table, "removeDBInfo")
      logger.info(`[${DATASET_CD}] Set DataSource DB Info `)
      await _exeQuery(C.Q.TABDATASET, data.table, "setDBInfo")
      _analysis(DATASET_CD, null, data)
    } else _analysis(DATASET_CD, filePaths, data)

    let insertData = []
    let backupData = []
    const spliceSize = 1000
    while (param.DATA.length > 0) {
      insertData.push(param.DATA.splice(0, spliceSize))
    }
    logger.info(
      `[${DATASET_CD}] Data Element Insert (${insertData.length} times)`
    )

    let spliceIdx = 0

    for (let i = 0; i < insertData.length; i++) {
      spliceIdx++
      logger.debug(
        `[${DATASET_CD}] Insert Data Element (${spliceIdx}/${insertData.length})`
      )
      param.DATA = insertData[i]
      backupData = backupData.concat(insertData[i])
      await _exeQuery(C.Q.DATASET, param, "setDataElement")
    }

    // param = {}
    // param.DATASET_CD = DATASET_CD
    // param.DATASET_STS = "DONE"
    // await _exeQuery(C.Q.DATASET, param, "updateDataSetStatus")
  } catch (error) {
    logger.error(`[${DATASET_CD}] Create Fail \n${error.stack}`)
    param = {}
    param.DATASET_CD = DATASET_CD
    param.DATASET_STS = "CRN_FAIL"
    param.LAST_MSG = error
    param.AUTO_ACC = data.AUTO_ACC
    await _exeQuery(C.Q.DATASET, param, "setUpdateDataset")
  }
}

const _analysis = async (code, filename, datasetInfo) => {
  let args = {}
  if (datasetInfo.UPLOAD_TYPE === "DB") {
    args.DB_INFO = {}
    datasetInfo.KEY = "WWED"
    args.DB_INFO = await _exeQuery(C.Q.TABDATASET, datasetInfo, "getDBInfo")
    args.DB_INFO = args.DB_INFO[0]
    args.DB_INFO.PASSWORD = String(args.DB_INFO.PASSWORD)
  } else {
    args = { FILE_PATH: filename }
  }
  logger.info(`[${code}] Starting EDA Analysis`)
  let data = await CF.runProcess("python", [
    C.BIN.runAnalysis,
    JSON.stringify(args)
  ]).catch((err) => {
    logger.error(`Analysis Fail \n${err.stderr}`)
  })
  logger.info(`[${code}] Success EDA Analysis !`)
  //ehdals
  try {
    if (data === undefined) throw new Error("EDA Data Undefined")
    if (data.STATUS === 0) throw new Error(data.MSG)
    let param = {}
    param.DATASET_CD = code
    param.DATASET_STS = "CREATE"
    await _exeQuery(C.Q.DATASET, param, "updateDataSetStatus")

    // let edaData = JSON.parse(data.stdout)
    let edaData = JSON.parse(data.stdout.replace(/\bNaN\b/g, "null"))
    if (edaData.STATUS === 0) throw new Error(edaData.MSG)

    param.DATASET_CD = code
    param.ROW_CNT = edaData.OVER_VIEW.DATASET_STATISTICS.COUNT
    param.COL_CNT = edaData.OVER_VIEW.DATASET_STATISTICS.VARIABLES_COUNT
    param.COLUMNS = "temp"
    param.COL_INFO = "temp"
    param.TARGET = "temp"
    param.SAMPLES = data.stdout
    param.CLASS_CNT = 0
    _exeQuery(C.Q.DATASET, param, "setAnalysis")

    param.DATASET_CD = code
    param.DATASET_STS = "DONE"
    await _exeQuery(C.Q.DATASET, param, "updateDataSetStatus")
  } catch (error) {
    let param = {}
    param.DATASET_CD = code
    param.DATASET_STS = "CRN_FAIL"
    await _exeQuery(C.Q.DATASET, param, "updateDataSetStatus")
    logger.error(`Analysis Fail \na${error}`)
  }
}

const _analysis_jogoon = async (code, filename, target) => {
  let info = {}
  return fs
    .createReadStream(filename)
    .pipe(
      parse({
        headers: (headers) => {
          info["columnCnt"] = headers.length
          info["columns"] = headers
          info["target"] = target ? target : headers[headers.length - 1]
          info["columnInfo"] = {}
          info["sampleRows"] = []
          info["rowCount"] = 0
          return headers
        }
      })
    )
    .on("error", (error) => console.error(error))
    .on("data", (row) => {
      // info.classes[row[info["target"]]] === undefined
      // 	? (info.classes[row[info["target"]]] = 1)
      // 	: (info.classes[row[info["target"]]] += 1)
      if (info["sampleRows"].length < 20) info["sampleRows"].push(row)
      info["rowCount"] += 1
      info.columns.map((column) => {
        // 컬럼 정보 초기화
        if (info.columnInfo[column] === undefined)
          info.columnInfo[column] = {
            missing: 0,
            unique: 0,
            type: Number(row[column]) ? "number" : "string",
            min: Number(row[column]) ? Number(row[column]) : row[column].length,
            max: Number(row[column]) ? Number(row[column]) : row[column].length,
            sum: 0,
            mean: 0,
            error: "",
            sorted: true,
            sortDirection: null,
            suggest: [],
            classes: {}
          }
        // 컬럼이 빈 값인 경우
        if (row[column] === null || row[column] === "")
          info.columnInfo[column].missing += 1
        else {
          let nData
          if (info.columnInfo[column].type === "number") {
            nData = Number(row[column])
          } else nData = row[column].length
          if (!nData)
            info.columnInfo[
              column
            ].error = `Data format is not unified ${typeof row[column]} ${
              info.columnInfo[column].type
            }`

          if (nData < info.columnInfo[column].min)
            info.columnInfo[column].min = nData
          else if (nData > info.columnInfo[column].max)
            info.columnInfo[column].max = nData

          info.columnInfo[column].sum += nData

          info.columnInfo[column].classes[row[column]] === undefined
            ? (info.columnInfo[column].classes[row[column]] = 1)
            : (info.columnInfo[column].classes[row[column]] += 1)

          info.columnInfo[column].unique = Object.keys(
            info.columnInfo[column].classes
          ).length

          info.columnInfo[column].mean =
            info.columnInfo[column].sum / info["rowCount"]
        }
      })
      // console.log(row)
    })
    .on("end", (rowCount) => {
      info.columns.map((column) => {
        if (info.columnInfo[column].unique === info.rowCount) {
          info.columnInfo[column].suggest.push(
            "Unique count is same as row count. It might be a primary key"
          )
          let prev = null
          if (info.columnInfo[column].type === "number") {
            Object.keys(info.columnInfo[column].classes).map((key) => {
              let nKey = Number(key)
              let curDirection = prev ? (nKey > prev ? "ASC" : "DESC") : "ASC"
              if (info.columnInfo[column].sortDirection === null)
                info.columnInfo[column].sortDirection = curDirection
              else {
                if (info.columnInfo[column].sortDirection !== curDirection)
                  info.columnInfo[column].sorted = false
              }
              prev = Number(key)
            })
          }
        }
        if (info.columnInfo[column].sorted)
          info.columnInfo[column].suggest.push(
            `The data values are contiguous(${info.columnInfo[column].sortDirection}).`
          )

        if (info.columnInfo[column].unique / info.rowCount > 0.1)
          info.columnInfo[column].suggest.push(
            `Data contains ${info.columnInfo[column].unique} values out of ${info.rowCount}. It might be categorical data`
          )
      })
      let param = {}
      param.DATASET_CD = code
      param.ROW_CNT = info.rowCount
      param.COL_CNT = info.columnCnt
      param.COLUMNS = JSON.stringify(info.columns)
      param.COL_INFO = JSON.stringify(info.columnInfo)
      param.TARGET = info.target
      param.SAMPLES = JSON.stringify(info.sampleRows)
      param.CLASS_CNT = info.columnInfo[info.target].unique
      _exeQuery(C.Q.DATASET, param, "setAnalysis")
      // dev
      // console.log(info)
      // console.log(`Parsed ${rowCount} rows`)
    })
}

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

router.post(
  "/getFileList",
  asyncHandler(async (req, res, next) => {
    let param = req.body
    let list = await _exeQuery(C.Q.DATASET, param, "getFileList")
    const result = list.map(
      (el) =>
        new Promise(async (resolve, reject) => {
          try {
            let data = await CF.runProcess("python", [
              C.BIN.getHeader,
              el.FILE_PATH
            ])
            el.columns = JSON.parse(data.stdout).COLUMNS
            resolve(el)
          } catch (err) {
            logger.error(`getHeader Fail \n${err.stderr}`)
            reject(err)
          }
        })
    )

    Promise.all(result)
      .then((r) => {
        res.json(r)
      })
      .catch((err) => {
        res.status(500).send({
          error: "getHeader Fail",
          msg: String(err.message)
        })
      })
  })
)
router.post("/getAnalysis", async (req, res, next) => {
  let param = req.body
  let data = await _exeQuery(C.Q.DATASET, param, "getAnalysis")
  res.json(data)
})

router.post("/dbconnection", async (req, res, next) => {
  // let DB = {
  //   "CLIENT": "oracledb",
  //   "ADDRESS": "106.251.247.178",
  //   "PORT": 9154,
  //   "DBNAME": "XE",
  //   "USER": "CTMSPLUS",
  //   "PASSWORD": "HKTIRE_CTMS"
  // }

  const dbType = req.body.CLIENT
  const conTest = req.body.IS_TEST
  let param = req.body

  param.include_limit = true
  if (conTest) {
    switch (dbType) {
      case "oracledb":
        param.QUERY = "SELECT 1 FROM DUAL"
        param.include_limit = false
        break

      case "db2":
        param.QUERY = "SELECT 1 FROM SYSIBM.SYSDUMMY1"
        break

      default:
        param.QUERY = "SELECT 1"
        break
    }
  }
  let result = await EXDB.extDBConnection(param)
  res.json(result)
})
module.exports = router
