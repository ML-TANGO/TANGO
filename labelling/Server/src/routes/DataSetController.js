import express from "express"
import asyncHandler from "express-async-handler"
import multer from "multer"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import fse from "fs-extra"
import rimraf from "rimraf"

import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"
var logger = require("../lib/Logger")(__filename)

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
    const fileList = JSON.parse(req.body.dir)

    let result = []
    fileList.map((ele) => {
      result.push({ path: ele, fileName: path.basename(ele), status: 1 })
    })

    res.json(result)
  })
)

router.post(
  "/removeTempFiles",
  asyncHandler(async (req, res, next) => {
    const tempDir = path.join(config.tempPath, req.body.uuid)
    const fileList = req.body.fileList
    fileList.map((ele) => {
      fse.removeSync(path.join(tempDir, ele.path))
    })
    res.json({ status: 1 })
  })
)

router.post(
  "/getDataSetList",
  asyncHandler(async (req, res, next) => {
    let thumUrl = "/static/"
    let option = {}
    option.source = CC.MAPPER.DATASET
    option.queryId = "getDataSetList"
    let list = await DH.executeQuery(option)

    let autoList = []
    list.map((ele) => {
      if (ele.DATASET_STS === "AUTO") autoList.push(ele)
      if (ele.THUM_NAIL !== null)
        ele.THUM_NAIL =
          thumUrl + ele.DATASET_CD + "/" + path.basename(ele.THUM_NAIL)
    })

    if (autoList.length > 0) {
      let failList = []
      let result = await CF.sendRequestResLong(
        config.masterIp,
        config.masterPort,
        CC.URI.activePredictor,
        {}
      )

      list.map((ele) => {
        if (ele.DATASET_STS === "AUTO") {
          let isRun = false

          if (result.length > 0)
            result.map((rpid) => {
              // jogoon 수정. ele.PID가 없어서 오토라벨링이 fail로 초기화됨
              // getDataSetList(xml)에서 PID가져오는 부분이 없어서 추가
              if (ele.PID === rpid.PID) isRun = true
            })

          if (!isRun) {
            logger.debug(
              `AUTO LABEL Fail ${JSON.stringify(result, null, 1)}  [${
                ele.DATASET_CD
              }]`
            )
            failList.push(ele.DATASET_CD)
            ele.DATASET_STS = "AUTO_FAIL"
            ele.FAIL_MSG = "오토라벨 프로세스 비정상 종료"
          }
        }
      })

      option.param = {}
      option.param.DATASET_STS = "AUTO_FAIL"
      option.param.FAIL_LIST = failList
      option.param.LAST_MSG = "Process does not Exist"
      option.queryId = "updaeFailDataSet"
      if (failList.length > 0) {
        await DH.executeQuery(option)
      }
    }

    res.json(list)
  })
)

router.post("/setUpdateDataset", async (req, res, next) => {
  console.log(req.body)
  let option = {}
  let rollback = {}
  rollback.orgData = {}
  rollback.newData = {}
  // req.body.AUTO_ACC = null
  if (req.body.AUTO_TYPE === "Y") {
    req.body.AUTO_EPOCH =
      req.body.AUTO_EPOCH === undefined ? -1 : req.body.AUTO_EPOCH
  }
  option.source = CC.MAPPER.DATASET
  option.queryId = "setUpdateDataset"
  option.param = req.body

  const datasetDir = path.join(config.datasetPath, req.body.DATASET_CD)
  const tempDir = path.join(config.tempPath, req.body.uuid)
  req.body.DATASET_DIR = datasetDir
  await DH.executeQuery(option)
  if (req.body.remove !== undefined && req.body.remove.length > 0) {
    let removeFiles = req.body.remove.map((file) => {
      return `'${path.parse(file).name}'`
    })
    removeFiles = removeFiles.join(",")

    option.queryId = "removeDataElementsByName"
    option.param.DATASET_CD = req.body.DATASET_CD
    option.param.FILE_NAMES = removeFiles
    await DH.executeQuery(option)
    // file System 삭제 mingi
    req.body.remove.map((removeFile) => {
      let fileName = path.join(datasetDir, removeFile)
      let dat = path.join(datasetDir, `${path.parse(removeFile).name}.dat`)
      let mask = path.join(
        datasetDir,
        `MASK_${path.parse(removeFile).name}.png`
      )
      fse.removeSync(fileName)
      fse.removeSync(dat)
      fse.removeSync(mask)
    })
  }

  //신규로직 
  let changed = []
  if(req.body.OBJECT_TYPE === "C") {
    const promises = req.body.files.map((ele) => {
      return new Promise((resolve, reject) => {
        if(ele.isNew) {
          try {
            let classDir = path.join(tempDir, ele.base)
            !fs.existsSync(classDir) && fs.mkdirSync(classDir)
    
            let imgPath = path.join(classDir, ele.path)
    
            fs.renameSync(path.join(tempDir, ele.path), imgPath)
            ele.path = path.join(ele.base, ele.path)
          } catch (error) {
            console.log("!!! isNew")
            console.log(error)
          }
        } else {

          if(!ele.path.includes(ele.base)) {
            try {

              let classDir = path.join(datasetDir, ele.base)
              !fs.existsSync(classDir) && fs.mkdirSync(classDir)
      
              let imgPath = path.join(classDir, ele.name)

              console.log("----Change")
              console.log(ele.path)

              console.log("----Change to")
              console.log(imgPath)
      
              fs.renameSync(ele.path, imgPath)
              ele.path = path.join(classDir, ele.name)

              changed.push(ele)
            } catch (error) {
              console.log("!!! ORG")
              console.log(error)
            }
          }
        }
        resolve("")
      })
    })

    await Promise.all(promises)    
  }

  console.log("---------------------")
  console.log(req.body)

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

  option.queryId = "getMaxDataCd"
  let dataCd = await DH.executeQuery(option)
  dataCd = Number(dataCd[0].DATA_CD) + 1

  option.source = CC.MAPPER.IMG_ANNO
  option.param = {}
  option.param.DATASET_CD = req.body.DATASET_CD
  option.param.DATASET_STS = "CREATE"
  option.queryId = "updateDataSetStatus"
  await DH.executeQuery(option)

  req.body.files = fileList
  // req.body.AUTO_TYPE = "N"
  req.body.changed = changed
  _createDataSets(req.body, dataCd)

  res.json({ status: 1 })
})

router.post(
  "/getFileList",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.DATASET
    option.queryId = "getFileList"
    option.param = req.body
    let list = await DH.executeQuery(option)
    res.json(list)
  })
)

router.post("/removeDataSet", async (req, res, next) => {
  let option = {}
  option.source = CC.MAPPER.DATASET
  option.queryId = "setUpdateDataset"
  option.param = req.body
  option.param.DATASET_STS = "DELETE"
  await DH.executeQuery(option)

  res.json({ status: 1 })
  _removeDataSet(req.body)
})

const _removeDataSet = async (data) => {
  try {
    logger.info(`Removing Dataset.... [${data.DATASET_CD}]`)
    const datasetDir = path.join(config.datasetPath, data.DATASET_CD)
    fs.existsSync(datasetDir) && rimraf.sync(datasetDir)
    let option = {}
    option.source = CC.MAPPER.DATASET
    option.queryId = "removeDataElements"
    option.param = data
    await DH.executeQuery(option)

    option.queryId = "removeAnalysis"
    await DH.executeQuery(option)

    option.queryId = "removeFeatures"
    await DH.executeQuery(option)

    option.queryId = "removeDataSet"
    await DH.executeQuery(option)
    logger.info(`Remove DataSet Success [${data.DATASET_CD}]`)
  } catch (error) {
    logger.error(`Remove Fail [${data.DATASET_CD}] \n${error.stack}`)

    let option = {}
    option.source = CC.MAPPER.DATASET
    option.queryId = "setUpdateDataset"
    option.param = data
    option.param.DATASET_STS = "DEL_FAIL"
    option.param.LAST_MSG = error.stack
    await DH.executeQuery(option)
  }
}

router.post("/testtest", async (req, res, next) => {
  console.log("test!")
  try {
    let resResult = await CF.sendGetRequest(
      "0.0.0.0",
      "10236",
      "/test",
        {
          project_id: "testtest",
          user_id: "dkdkdkfekjfj"
        }
    )
    console.log("====")
    console.log(resResult)
    console.log("====")
  } catch (error) {
    console.log("error")
    console.log(error)
  }
})

router.post("/setDupDataset", async (req, res, next) => {
  let option = {}
  const tempDir = path.join(config.datasetPath, req.body.ORG_DATASET_CD)
  option.source = CC.MAPPER.DATASET
  option.queryId = "setNewDataSetNumber"
  option.param = req.body
  option.param.YEAR = moment().format("YY")

  //신규 데이터셋 코드 생성
  option.param.ORG_DATASET_CD = req.body.ORG_DATASET_CD
  await DH.executeQuery(option)
  option.queryId = "getNewDataSetNumber"
  //신규 데이터셋 조회
  let datasetNo = await DH.executeQuery(option)
  datasetNo = datasetNo[0].DATASET_NUMBER

  const datasetDir = path.join(config.datasetPath, datasetNo)
  !fs.existsSync(datasetDir) && fs.mkdirSync(datasetDir)

  // 신규 Dataset 폴더로 복사
  fse.copySync(tempDir, datasetDir)

  let fileList = req.body.files

  //데이터셋 메인 썸네일 생성
  const img = fileList[0].path
  let thum = "THUM_" + path.parse(fileList[0].path).name + ".jpg"
  thum = path.join(datasetDir, thum)

  let resResult
  try {
    resResult = await CF.sendRequestResLong(
      config.masterIp,
      config.masterPort,
      CC.URI.makeThumnail,
      [
        {
          MDL_TYPE: req.body.DATA_TYPE,
          PATH: img,
          SAVE_PATH: thum
        }
      ]
    )
    if (resResult.STATUS === 0) throw resResult.ERROR_FILE

    option.param.DATA = []
    option.param.DATA.push({
      DATASET_CD: datasetNo,
      DATA_CD: "T0000000",
      DATA_STATUS: "THUM",
      FILE_NAME: path.parse(thum).name,
      FILE_EXT: path.parse(thum).ext,
      FILE_TYPE: req.body.DATA_TYPE,
      FILE_PATH: thum,
      FILE_RELPATH: path.basename(thum),
      FILE_SIZE: 1,
      FPS: 0,
      TAG_CD: 0
    })

    option.queryId = "setDataElement"
    await DH.executeQuery(option)
  } catch (error) {
    logger.error(error.message)
    res.json({ status: 0, err: error, msg: "썸네일 생성 실패" })
    return
  }
  option.param.DATASET_CD = datasetNo
  option.param.CRN_USR = req.body.USER_ID || "user"
  option.param.THUM_NAIL_CD = "T0000000"
  option.param.CATEGORY1 = "USER"
  option.queryId = "setNewDataSet"
  await DH.executeQuery(option)

  option.source = CC.MAPPER.IMG_ANNO
  option.param = {}
  option.param.DATASET_CD = datasetNo
  option.param.DATASET_STS = "CREATE"
  option.queryId = "updateDataSetStatus"
  await DH.executeQuery(option)

  logger.debug("File Path Substr Start")
  req.body.files.map((ele) => {
    ele.path = ele.path.substr(datasetDir.length, ele.path.length)
  })
  logger.debug("File Path Substr Done")
  req.body.DATASET_DIR = datasetDir
  _createDataSets(req.body, 0)

  res.json({ status: 1 })
})

router.post("/setNewDataSets", async (req, res, next) => {

  let option = {}
  let rollback = {}
  rollback.orgData = {}
  rollback.newData = {}

  const tempDir = path.join(config.tempPath, req.body.uuid)
  option.source = CC.MAPPER.DATASET
  option.queryId = "setNewDataSetNumber"
  option.param = req.body
  option.param.YEAR = moment().format("YY")

  //신규 데이터셋 코드 생성
  await DH.executeQuery(option)
  option.queryId = "getNewDataSetNumber"
  //신규 데이터셋 조회
  let datasetNo = await DH.executeQuery(option)
  datasetNo = datasetNo[0].DATASET_NUMBER
  req.body.DATASET_CD = datasetNo

  const datasetDir = path.join(config.datasetPath, datasetNo)
  req.body.DATASET_DIR = datasetDir

  if(req.body.OBJECT_TYPE === "C") {
      //신규로직 
      const promises = req.body.files.map((ele) => {
        return new Promise((resolve, reject) => {

          if(!ele.path.includes(ele.base)) {
            try {
              let classDir = path.join(tempDir, ele.base)
              !fs.existsSync(classDir) && fs.mkdirSync(classDir)
  
              let imgPath = path.join(classDir, ele.path)
  
              fs.renameSync(path.join(tempDir, ele.path), imgPath)
              ele.path = path.join(ele.base, ele.path)
            } catch (error) {
              console.log("!!!")
              console.log(error)
            }
          }
          resolve("")
        })
      })

      await Promise.all(promises)
  }

  try {
    // !fs.existsSync(datasetDir) && fs.mkdirSync(datasetDir)
    logger.info(`${tempDir}    ->>     ${datasetDir}`)
    console.log(tempDir)
    console.log(fs.existsSync(tempDir))
    console.log(datasetDir)
    console.log(fs.existsSync(datasetDir))
    fs.renameSync(tempDir, datasetDir)
    let resultPath = path.join(datasetDir, "result")
    !fs.existsSync(resultPath) && fs.mkdirSync(resultPath)
  } catch (error) {
    logger.error(error.message)
    res.json({ status: 0, err: error.message, msg: "폴더생성실패" })
    return
  }

  logger.info(`Temp file copy Done ${datasetDir}`)
  let fileList = req.body.files
  fileList.map((ele) => {
    ele.path = ele.path.replace(/ /g, "")
  })

  rollback.orgData.files = fileList
  rollback.orgData.tempPath = tempDir
  rollback.newData.dataPath = datasetDir

  //데이터셋 메인 썸네일 생성
  let imgIdx = 0
  if (req.body.IMPORT_TYPE === "COCO") imgIdx = 1
  const img = path.join(datasetDir, fileList[imgIdx].path)
  let thum = "THUM_" + path.parse(fileList[imgIdx].path).name + ".jpg"
  thum = path.join(datasetDir, thum)

  let resResult
  try {
    resResult = await CF.sendRequestResLong(
      config.masterIp,
      config.masterPort,
      CC.URI.makeThumnail,
      [
        {
          MDL_TYPE: req.body.DATA_TYPE,
          PATH: img,
          SAVE_PATH: thum
        }
      ]
    )
    if (resResult.STATUS === 0) throw resResult.ERROR_FILE

    option.param.DATA = []
    option.param.DATA.push({
      DATASET_CD: datasetNo,
      DATA_CD: "T0000000",
      DATA_STATUS: "THUM",
      FILE_NAME: path.parse(thum).name,
      FILE_EXT: path.parse(thum).ext,
      FILE_TYPE: req.body.DATA_TYPE,
      FILE_PATH: thum,
      FILE_RELPATH: path.basename(thum),
      FILE_SIZE: 1,
      FPS: 0,
      TAG_CD: 0
    })

    option.queryId = "setDataElement"
    await DH.executeQuery(option)
  } catch (error) {
    logger.error(error)
    res.json({ status: 0, err: error, msg: "썸네일 생성 실패" })
    rollbackDataSet(rollback.orgData, rollback.newData)
    return
  }
  option.param.DATASET_CD = datasetNo
  option.param.CRN_USR = req.body.USER_ID || "user"
  option.param.THUM_NAIL_CD = "T0000000"
  option.param.CATEGORY1 = "USER"
  option.queryId = "setNewDataSet"
  await DH.executeQuery(option)

  option.source = CC.MAPPER.IMG_ANNO
  option.param = {}
  option.param.DATASET_CD = datasetNo
  option.param.DATASET_STS = "CREATE"
  option.queryId = "updateDataSetStatus"
  await DH.executeQuery(option)

  _createDataSets(req.body, 0)


  res.json({ status: 1 })

  
  option.source = CC.MAPPER.TANGO
  option.queryId = "getProjectInfo"
  let list = await DH.executeQuery(option)
  
  
  if(list[0] !== undefined) {
    logger.info("send status request to Project Manager")
    console.log(list[0])
    try {

      let resResult = await CF.sendGetRequest(
        "project_manager",
        "8085",
        "/status_request",
          {
            project_id: list[0].PROJECT_ID,
            user_id: list[0].USER_ID
          }
      )
    } catch (error) {
      console.log("error")
      console.log(error)
    }
    
  } else
  logger.info("Fail to send status request to Project Manager : not started")
  
  // var os = require('os');


  // // http://:project_manager:8085)

  // var networkInterfaces = os.networkInterfaces();

  // console.log(networkInterfaces);
  // logger.info("IP INFO")
  // logger.info(networkInterfaces)
  

  
})

const _createDataSets = async (data, dataCd) => {
  logger.info("DataSet Create Start")
  let fileList = data.files
  const datasetDir = data.DATASET_DIR
  const DATASET_CD = data.DATASET_CD

  let option = {}
  option.source = CC.MAPPER.DATASET
  option.param = data

  try {
    //비디오 데이터 썸네일 생성
    if (data.DATA_TYPE === "V") {
      logger.info("Video Data Processing...")
      let tmpFile = []
      fileList.map((ele) => {
        tmpFile.push({
          PATH: path.join(datasetDir, ele.path),
          FILE_NAME: ele.name
        })
      })
      let videoInfo = await CF.sendRequestResLong(
        config.masterIp,
        config.masterPort,
        CC.URI.videoInfo,
        tmpFile
      )
      if (videoInfo.STATUS === 0) {
        logger.error(videoInfo)
        return
      }

      videoInfo.map((ele) => {
        fileList.map((file) => {
          if (ele.FILE_NAME === file.name) file.FPS = ele.FPS
        })
      })
      //비디오 일 경우 썸네일 전체 생성
      const promises = fileList.map((ele) => {
        return new Promise((resolve, reject) => {
          const elePath = path.join(datasetDir, ele.path)
          let video = "THUM_" + path.parse(ele.path).name + ".jpg"
          video = path.join(datasetDir, video)
          const result = CF.sendRequestResLong(
            config.masterIp,
            config.masterPort,
            CC.URI.makeThumnail,
            [
              {
                MDL_TYPE: data.DATA_TYPE,
                PATH: elePath,
                SAVE_PATH: video
              }
            ]
          ).catch((err) => {
            reject(err)
          })
          resolve(result)
        })
      })

      await Promise.all(promises)
    } else if (data.DATA_TYPE === "I") {
      let saveTumPath = data.DATASET_DIR
      let thumFiles = []

      data.files.map((ele) => {
        let filePath = data.DATASET_DIR
        filePath = path.join(filePath, ele.path)
        let savePath = path.dirname(filePath)
        savePath = path.join(savePath, "THUM_" + ele.name)
        savePath = savePath.replace(/ /g, "")
        filePath = filePath.replace(/ /g, "")
        thumFiles.push({
          MDL_TYPE: data.DATA_TYPE,
          PATH: filePath,
          SAVE_PATH: savePath
        })
      })

      const result = await CF.sendRequestResLong(
        config.masterIp,
        config.masterPort,
        CC.URI.makeThumnail,
        thumFiles
      ).catch((err) => {
        throw new Error(`Thumnail Network Fail ${err.stack}`)
      })
      // 썸네일 실패 파일이 포함되더라도 생성은 진행
      // if (result.STATUS === 0) throw new Error("Thumnail Fail")
    }
    //클레시피케이션 클래스 생성
    let TAG_CD_LIST = []
    if (data.OBJECT_TYPE === "C") {
      logger.info("Classification Data Processing...")
      const classList = fs.readdirSync(datasetDir)

      option.queryId = "getExistTag"
      let existTagList = await DH.executeQuery(option)
      console.log("==========================1")
      console.log(existTagList)
      option.param.TAG = []

      console.log("==========================1-1")
      console.log(classList)

      classList.map((ele) => {
        let existTag = existTagList.findIndex((tagname) => tagname.NAME === ele)

        console.log(existTag)
        console.log(path.parse(ele).ext)

        if (existTag < 0 && ele !== "result" && path.parse(ele).ext === "") {
        // if (existTag < 0 && ele !== "result" && ele.substring(0,4) !== "THUM") {
          option.param.TAG.push({
            DATASET_CD: DATASET_CD,
            NAME: ele,
            CLASS_SUFFIX: ele
          })
        }
      })


      option.queryId = "setNewTag"
      if (option.param.TAG.length > 0) await DH.executeQuery(option)
      option.param.DATASET_CD = DATASET_CD
      option.queryId = "getExistTag"
      TAG_CD_LIST = await DH.executeQuery(option)


      if(data.changed !== undefined && data.changed.length > 0) {
        let DATAS = []

        data.changed.map(ele => {
          let tempData = {}
          tempData.FILE_NAME = path.parse(ele.name).name
          tempData.FILE_PATH = ele.path
          let changedTag = TAG_CD_LIST.filter(tag => tag.NAME === ele.base)
          tempData.TAG_CD = changedTag[0].TAG_CD
          DATAS.push(tempData)
        })

        let optionTemp = {}
        optionTemp.source = CC.MAPPER.DATASET
        optionTemp.param = {
          DATASET_CD : DATASET_CD,
          DATAS
        }
        console.log(DATAS)
        optionTemp.queryId = "setBulkUpdateDataElement"
        await DH.executeQuery(optionTemp)
      }
    }

    option.param.DATA = []
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
      if (data.OBJECT_TYPE === "C") {
        const className = path.basename(path.dirname(ele.path))
        const tag = TAG_CD_LIST.find((item) => item.NAME === className)
        logger.info(`find TagCD : ${JSON.stringify(tag)}`)
        tempEle.TAG_CD = tag.TAG_CD
      }

      option.param.DATA.push(tempEle)
    })


    let insertData = []
    let backupData = []
    const spliceSize = 1000
    while (option.param.DATA.length > 0) {
      insertData.push(option.param.DATA.splice(0, spliceSize))
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
      option.param.DATA = insertData[i]
      backupData = backupData.concat(insertData[i])
      option.queryId = "setDataElement"
      await DH.executeQuery(option)
    }

    //데이터 임폴트인 경우
    if (data.IMPORT_TYPE === "COCO") {
    let argData = {
      "DATASET_CD": DATASET_CD,
      "PURPOSE_TYPE": data.OBJECT_TYPE,
      "FILE_INFO":[]
    }

    data.DATA.map(ele => {
      argData.FILE_INFO.push({"FILE_PATH": ele.FILE_PATH, "DATA_CD":ele.DATA_CD})
    })

    logger.info("Import Data Processing...")
    let importResult = await CF.runProcess("python", [
      CC.BIN.importData,
      JSON.stringify(argData)
    ])

    if(importResult.stderr === "") {
      importResult = JSON.parse(importResult.stdout)
      
      const classList = importResult.CLASS_INFO


      let importOption = {}
      importOption.param = {}
      importOption.source = CC.MAPPER.BIN
      importOption.param.DATASET_CD = DATASET_CD
      importOption.param.DATA = []
      classList.map((ele) => {
        importOption.param.DATA.push({
            DATASET_CD: data.DATASET_CD,
            NAME: ele.TAG_NAME,
            COLOR: ele.COLOR
          })
      })
      importOption.queryId = "setDataTags"
      if (importOption.param.DATA.length > 0) {
        await DH.executeQuery(importOption)

        // param.map((ele) => {
        //   tagDatas = tagDatas.concat(ele.TAGS)
        //   option.param.DATA.push({
        //     DATA_CD: ele.DATA_CD,
        //     ANNO_DATA: ele.FILE_PATH
        //   })
        // })
        importOption.source = CC.MAPPER.BIN
        importOption.param.DATA = []
        data.DATA.map(ele => {
          let fn = ele.FILE_PATH
          let ext = ele.FILE_EXT


          let an = fn.substr(0, fn.length-ext.length)
          an = an+".dat"

          if(ext !== ".json") {
            importOption.param.DATA.push({
              DATA_CD: ele.DATA_CD,
              ANNO_DATA: an
            })
          }
        })

        //모든 태그 변경

        importOption.source = CC.MAPPER.DATASET
        importOption.queryId = "getExistTag"
        let existTagList = await DH.executeQuery(importOption)

        

        existTagList.map(ele => {
          let findTag = classList.find((item) => item.TAG_NAME === ele.NAME)
          ele.orgTagCd = findTag.TAG_CD
        })

        logger.info("Imported CLass Data Mapping...")
        const chFile = importOption.param.DATA.map((ele) => {
          return new Promise((resolve, reject) => {
            let filePath = ele.ANNO_DATA
            if (fs.existsSync(filePath)) {
              var rsFile = JSON.parse(String(fs.readFileSync(filePath)))
              rsFile.POLYGON_DATA.map((frame) => {

                if (frame.TAG_CD != undefined && frame.TAG_CD != null) {

                  let findTag = existTagList.find((item) => item.orgTagCd === frame.TAG_CD)
                  console.log(`findeTag = ${frame.TAG_CD}  to ${findTag}`)
                  frame.TAG_CD = findTag.TAG_CD
                }
              })
              fs.writeFileSync(filePath, JSON.stringify(rsFile), {
                encoding: "utf8",
                flag: "w"
              })
            }
            resolve(1)
          })
        })
    
        await Promise.all(chFile)
        logger.info("Imported CLass Data Mapping Done!")




        //데이터 엘리먼트 수정
        importOption.source = CC.MAPPER.BIN
        importOption.param.IS_ANNO = true
        importOption.queryId = "setUpdateDataElement"
        await DH.executeQuery(importOption)

        importOption.queryId = "removeJson"
        await DH.executeQuery(importOption)
      }
    }
    

    }

    if (data.AUTO_TYPE === "Y") {
      data.AUTO_EPOCH = data.EPOCH === undefined ? -1 : data.EPOCH
      option.source = CC.MAPPER.DATASET
      option.param = {}
      option.param.DATASET_CD = DATASET_CD
      option.param.AUTO_ACC = data.AUTO_ACC
      option.param.AUTO_MODEL = data.AUTO_MODEL
      option.param.AUTO_EPOCH = data.AUTO_EPOCH
      option.param.AUTO_TYPE = data.AUTO_TYPE
      option.queryId = "setUpdateDataset"
      await DH.executeQuery(option)

      await _prePredictAll(DATASET_CD, data, data.AUTO_ACC)
    } else {
      option.source = CC.MAPPER.IMG_ANNO
      option.param = {}
      option.param.DATASET_CD = DATASET_CD
      option.param.DATASET_STS = "DONE"
      option.queryId = "updateDataSetStatus"
      await DH.executeQuery(option)
    }


    
  } catch (error) {
    logger.error(`[${DATASET_CD}] Create Fail \n${error.stack}`)
    option.source = CC.MAPPER.DATASET
    option.param = {}
    option.param.DATASET_CD = DATASET_CD
    option.param.DATASET_STS = "CRN_FAIL"
    option.param.LAST_MSG = error
    option.param.AUTO_ACC = data.AUTO_ACC
    option.param.AUTO_MODEL = data.AUTO_MODEL
    option.param.AUTO_EPOCH = data.AUTO_EPOCH
    option.param.AUTO_TYPE = data.AUTO_TYPE
    option.queryId = "setUpdateDataset"
    await DH.executeQuery(option)
  }
}

const rollbackDataSet = (orgData, newData) => {
  try {
    logger.info("Rollback Dataset")
    !fs.existsSync(orgData.tempPath) && fs.mkdirSync(orgData.tempPath)
    orgData.files.map((ele) => {
      let orgFilePath = path.join(orgData.tempPath, ele.name)
      let newFilePath = path.join(newData.dataPath, ele.name)
      orgFilePath = orgFilePath.replace(/ /g, "")
      newFilePath = newFilePath.replace(/ /g, "")
      !fs.existsSync(orgFilePath) && fs.renameSync(newFilePath, orgFilePath)
    })
  } catch (error) {
    logger.error(`[ ${orgData.tempPath} ] Rollbal Fail`)
  }
}

// jogoon 추가함. 로직은 확인해봐야함
// auto labeling 실패시 호출
router.post("/autolabeling", async (req, res, next) => {
  let option = {}
  option.source = CC.MAPPER.DATASET
  option.param = req.body
  // option.queryId = "getFileList"
  // let data = await DH.executeQuery(option)

  option.queryId = "getAutoLabelInfo"
  let autoLabelInfo = await DH.executeQuery(option)
  autoLabelInfo = autoLabelInfo[0]
  res.json({ status: 1 })
  _prePredictAll(option.param.DATASET_CD, autoLabelInfo, autoLabelInfo.AUTO_ACC)
})

const _prePredictAll = async (DATASET_CD, reqData, autoAcc) => {
  let option = {}
  let mdlPid = {}
  option.source = CC.MAPPER.IMG_ANNO
  option.param = {}
  option.param.DATASET_CD = DATASET_CD
  option.param.DATA_STATUS = "ORG"
  option.queryId = "getImageList"
  let dataEle = await DH.executeQuery(option)

  option.source = CC.MAPPER.DATASET
  option.param.DATASET_CD = DATASET_CD
  option.param.DATASET_STS = "AUTO"
  option.param.AUTO_ACC = autoAcc

  let imageInfo = []
  dataEle.map((ele) => {
    if (ele.DATA_STATUS === "ORG")
      imageInfo.push({
        IMAGE_PATH: ele.FILE_PATH,
        DATASET_CD: ele.DATASET_CD, //바이페스
        DATA_CD: ele.DATA_CD, //바이페스
        COLOR: null,
        RECT: null
      })
  })
  const OBJECT_TYPE = option.param.DATASET_CD.substr(0, 1)
  let param = {
    IMAGES: imageInfo,
    MODE: 2, //전체 예측 후 REQ던짐
    URL: "prePredict",
    OBJECT_TYPE: OBJECT_TYPE,
    // AI_CD: OBJECT_TYPE === "D" ? "YOLOV3" : "DEEP-LAB",
    AI_CD: reqData.AUTO_MODEL,
    CLASS_CD: "BASE",
    DATA_TYPE: dataEle[0].FILE_TYPE,
    EPOCH: reqData.AUTO_EPOCH,
    AUTO_ACC: autoAcc
  }

  //EPOCH가 -1이면 베이스 모델.
  if (param.EPOCH !== -1) {
    param.MDL_PATH = CF.getMdlPath(param.AI_CD, param.EPOCH)
  } else {
    option.source = CC.MAPPER.AIPRJ
    option.param.OBJECT_TYPE = OBJECT_TYPE
    option.param.DATA_TYPE = dataEle[0].FILE_TYPE
    option.param.NETWORK_NAME = reqData.AUTO_MODEL
    option.queryId = "getBaseModelByObjectType"
    let autoModel = await DH.executeQuery(option)
    autoModel = autoModel[0]

    param.MDL_PATH = autoModel.NETWORK_PATH
    option.source = CC.MAPPER.DATASET
  }

  if (param.DATA_TYPE === "V") param.URL = "prePredictVideo"
  let result = { status: 1 }
  try {
    mdlPid = await CF.sendRequestResLong(
      config.masterIp,
      config.masterPort,
      CC.URI.modelLoad,
      param
    )
    if (mdlPid.PID === 0) throw new Error("Model Load Fail")
    CF.sendRequest(
      config.masterIp,
      config.masterPort,
      CC.URI.autoLabeling,
      param
    )
  } catch (error) {
    logger.error(`[${DATASET_CD}] Auto Labeling Fail \n${error.stack}`)
    mdlPid.PID = null
    option.param.DATASET_STS = "AUTO_FAIL"
    option.param.LAST_MSG = error.stack
    option.param.AUTO_ACC = autoAcc
  } finally {
    option.param.PID = mdlPid.PID
    option.queryId = "setUpdateDataset"
    await DH.executeQuery(option)
    return mdlPid.PID
  }
}

router.post(
  "/getDataSetImportList",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.DATASET
    option.queryId = "getDataSetList"
    option.param = req.body
    let list = await DH.executeQuery(option)

    let sampleDaraSets = []
    let sampleDaraSets2 = []

    let dataSetTree = []
    dataSetTree.push({
      key: "SAMPLE",
      label: "SAMPLE DATA",
      selectable: false,
      nodes: []
    })
    dataSetTree.push({
      key: "USER",
      label: "USER DATASET",
      nodes: []
    })

    list.map((ele) => {
      ele.key = ele.DATASET_CD
      ele.label = ele.TITLE
      ele.selectable = true
      if (ele.CATEGORY1 === "USER") dataSetTree[1].nodes.push(ele)
      else if (ele.CATEGORY1 === "SAMPLE") sampleDaraSets.push(ele)
    })

    sampleDaraSets.map((ele, idx) => {
      if (Object.keys(sampleDaraSets).equal !== ele.CATEGORY2) {
        sampleDaraSets2[ele.CATEGORY2] = []
        sampleDaraSets2[ele.CATEGORY2].push(ele)
      } else sampleDaraSets2[ele.CATEGORY2].push(ele)
    })

    Object.keys(sampleDaraSets2).map((ele) => {
      dataSetTree[0].nodes.push({
        key: ele,
        label: ele,
        nodes: sampleDaraSets2[ele],
        selectable: false
      })
    })

    res.json(dataSetTree)
  })
)

router.post("/getBaseModel", async (req, res, next) => {
  let option = {}
  let models = []
  let data
  option.source = CC.MAPPER.DATASET
  option.param = req.body
  option.queryId = "getBaseModel"
  data = await DH.executeQuery(option)
  data.map((el) => {
    models.push({
      TYPE: "BASE",
      NAME: el.NETWORK_NAME,
      AI_CD: el.NETWORK_NAME,
      DATA_TYPE: el.DATA_TYPE,
      OBJECT_TYPE: el.OBJECT_TYPE
    })
  })
  option.queryId = "getUserModel"
  data = await DH.executeQuery(option)
  data.map((el) => {
    models.push({
      TYPE: "USER",
      NAME: el.TITLE,
      AI_CD: el.AI_CD,
      DATA_TYPE: el.DATA_TYPE,
      OBJECT_TYPE: el.OBJECT_TYPE
    })
  })
  res.json(models)
})

router.post("/getModelEpochs", async (req, res, next) => {
  let option = {}
  option.source = CC.MAPPER.DATASET
  option.param = req.body
  option.queryId = "getModelEpochs"
  let data = await DH.executeQuery(option)
  res.json(data)
})

router.post("/getExistTag", async (req, res, next) => {
  let option = {}
  option.source = CC.MAPPER.DATASET
  option.param = req.body
  option.queryId = "getExistTag"
  let data = await DH.executeQuery(option)
  res.json(data)
})

const countUnique = (arr) => {
  return arr.reduce(function (acc, curr) {
    if (typeof acc[curr] == "undefined") {
      acc[curr] = 1
    } else {
      acc[curr] += 1
    }

    return acc
  }, {})
}

router.post("/getTagInfo", async (req, res, next) => {
  // get image list
  let option = {}
  option.source = CC.MAPPER.IMG_ANNO
  option.queryId = "getImageList"
  option.param = req.body
  option.param.DATA_STATUS = "ORG"

  let list = await DH.executeQuery(option)
  let tagInfo = {}

  list.map((el) => {
    console.log(el.FILE_PATH)

    if (el.ANNO_DATA !== null && el.ANNO_DATA !== "") {
      if (path.extname(el.ANNO_DATA) === ".dat") {
        el.ANNO_DATA = fs.readFileSync(el.ANNO_DATA)
        let tempAnno = JSON.parse(el.ANNO_DATA)
        let tags = countUnique(tempAnno.POLYGON_DATA.map((ele) => ele.TAG_NAME))
        Object.keys(tags).map((key) => {
          tagInfo[key] = tagInfo[key] ? tagInfo[key] + tags[key] : tags[key]
        })
        console.log(tags)
      }
    }
  })

  // option.source = CC.MAPPER.DATASET
  // option.param = req.body
  // option.queryId = "getTagInfo"
  // let data = await DH.executeQuery(option)
  res.json(tagInfo)
})

module.exports = router
