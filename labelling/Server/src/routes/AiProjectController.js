import express from "express"
import asyncHandler from "express-async-handler"
import moment from "moment"
import fs from "fs"
import path, { basename } from "path"
import mime from "mime"
import archiver from "archiver"
import rimraf from "rimraf"
import getSize from "get-folder-size"

import DH from "../lib/DatabaseHandler"
import C from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"
import { list } from "tar"
const logger = require("../lib/Logger")(__filename)

const router = express.Router()
const CRN_USR = "testUser@test.co.kr"
const config = CL.getConfig()
const spawn = require("child_process").spawn

router.post(
  "/testConnection",
  asyncHandler(async (req, res, next) => {
    CF.sendSoc({ AI_CD: req.body.AI_CD, MSG: req.body.MSG }, "TRAIN")
  })
)
router.post(
  "/getAiPrjList",
  asyncHandler(async (req, res, next) => {
    let thumUrl = "/static"
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.param = req.body
    option.queryId = "getAiPrjList"
    let AiList = await DH.executeQuery(option)
    AiList.map((ele) => {
      if (ele.THUM_NAIL !== null) {
        let FILE_PATH
        const temp = ele.THUM_NAIL.indexOf(config.datasetPath)
        if (temp > -1) {
          FILE_PATH = ele.THUM_NAIL.substr(
            config.datasetPath.length + temp,
            ele.THUM_NAIL.length
          )
          ele.THUM_NAIL = thumUrl + FILE_PATH
        } else ele.THUM_NAIL = null
      }
    })
    res.json(AiList)
  })
)

router.post(
  "/getTrainingInfo",
  asyncHandler(async (req, res, next) => {
    res.json({
      SOCKET: `http://${config.ip}:${config.soc_port}`
    })
  })
)

router.post(
  "/setAiPrj",
  asyncHandler(async (req, res, next) => {
    const aiInfo = req.body.AI_INFO
    const dataSetInfo = req.body.DATASET
    const classInfo = req.body.CLASSES
    const isTransfer = aiInfo.CFG_TRANSFER === "yes"
    const isAuto = aiInfo.CFG_MODE === "Auto"
    const isEarlyStop = aiInfo.CFG_EARLY_STOP === "Y"

    req.body.USER_ID = aiInfo.USER_ID || "user"

    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.param = {}
    option.queryId = "setNewPrjNumber"
    option.param = aiInfo
    option.param.YEAR = moment().format("YYYY")

    ///////////채번/////////////////
    await DH.executeQuery(option)
    option.queryId = "getNewPrjNumber"
    let prjNumber = await DH.executeQuery(option)
    prjNumber = prjNumber[0].PRJ_NUMBER

    ////////AI 생성//////////////
    option.param.AI_CD = prjNumber
    option.param.AI_TYPE = aiInfo.OBJECT_TYPE + aiInfo.DATA_TYPE
    option.param.CRN_USR = req.body.USER_ID || "user"
    option.param.PATH = path.join(config.aiPath, prjNumber)
    option.param.NETWORK_NAME =
      aiInfo.CFG_BASEMODEL?.NETWORK_NAME !== undefined
        ? aiInfo.CFG_BASEMODEL.NETWORK_NAME
        : null
    option.queryId = "setAiPrj"
    await DH.executeQuery(option)

    ////////AI 폴더 생성///////////////
    !fs.existsSync(path.join(config.aiPath, String(prjNumber))) &&
      fs.mkdirSync(path.join(config.aiPath, String(prjNumber)), {
        recursive: true
      })

    //////데이터셋 삽입//////////
    option.param.DATA = []
    dataSetInfo.map((ele) => {
      option.param.DATA.push({ DATASET_CD: ele.dataset_cd })
    })
    option.queryId = "setTrainDataset"
    await DH.executeQuery(option)

    //////클래스 삽입 삽입//////////
    option.param.DATA = []
    let uniq = classInfo
      .map((label) => label.class)
      .filter((v, i, s) => s.indexOf(v) === i)
    uniq.map((ele) => {
      let temp = classInfo.filter((v) => v.class === ele)
      let tagElement = ""
      temp.map((item) => {
        tagElement += item.tag_cd + ","
      })
      tagElement = tagElement.substr(0, tagElement.length - 1)
      option.param.DATA.push({
        AI_CD: prjNumber,
        NAME: ele,
        DESC_TXT: null,
        COLOR: temp[0].color,
        ELEMENT_TAGS: tagElement
      })
    })
    option.queryId = "setTrainClassInfo"
    await DH.executeQuery(option)

    option.param = {}
    if (isTransfer) {
      let sourceAI_CD = aiInfo.CFG_SOURCEMODEL.AI_CD
      let sourceEpoch = aiInfo.CFG_SOURCEMODEL_EPOCH
      let mdl_path = CF.getMdlPath(aiInfo.CFG_SOURCEMODEL.AI_CD, sourceEpoch)

      option.param.BASE_AI_CD = sourceAI_CD
      option.param.NETWORK_PATH = mdl_path
      option.param.NETWORK_NAME = aiInfo.CFG_SOURCEMODEL.NETWORK_NAME
    } else if (!isAuto) {
      option.param.BASE_AI_CD = null
      option.param.NETWORK_NAME = aiInfo.CFG_BASEMODEL.NETWORK_NAME
      option.param.NETWORK_PATH = aiInfo.CFG_BASEMODEL.NETWORK_PATH
    }

    option.param.AI_CD = prjNumber
    option.param.IS_AUTO = aiInfo.CFG_MODE === "Auto"
    option.param.EPOCH = Number(aiInfo.CFG_EPOCH)
    option.param.BATCH_SIZE = Number(aiInfo.CFG_BATCH)
    option.param.NETWORK_CD = 0
    option.param.ACTIVE_FUNC = aiInfo.CFG_ACTIVE
    option.param.OPTIMIZER = aiInfo.CFG_OPTIMIZER
    option.param.LOSS_FUNC = aiInfo.CFG_LOSS
    option.param.IS_TRANSFER = isTransfer
    option.param.GPU_INDEX = JSON.stringify(aiInfo.CFG_GPUIDX)
    option.param.GPU_LIMIT = aiInfo.CFG_GPU
    option.param.MAX_TRIAL = null
    option.param.IS_EARLYSTOP = isEarlyStop
    option.param.EARLY_MONITOR = null
    option.param.EARLY_MODE = null
    option.param.IMG_SIZE = aiInfo.CFG_IMG_SIZE
    option.param.IMG_CHANNEL = aiInfo.CFG_IMG_CHANNEL

    // //테스트시 발견된 오류
    // option.param.NETWORK_NAME = aiInfo.CFG_BASEMODEL.NETWORK_NAME
    // option.param.NETWORK_PATH = aiInfo.CFG_BASEMODEL.NETWORK_PATH

    if (isAuto) {
      option.param.OBJECT_TYPE = aiInfo.OBJECT_TYPE
      option.param.DATA_TYPE = aiInfo.DATA_TYPE
      option.param.PIRIORITY = 1
      option.queryId = "getBaseModelByObjectType"
      let autoModel = await DH.executeQuery(option)

      // AUTOMODEL YOLO 넘어옴. 쿼리 수정 필요 - smpark
      autoModel = autoModel[0]
      console.log(aiInfo.OBJECT_TYPE)
      console.log(aiInfo.DATA_TYPE)
      console.log("==================")
      console.log(autoModel)
      option.param.NETWORK_CD = 0
      option.param.ACTIVE_FUNC = ""
      option.param.OPTIMIZER = ""
      option.param.LOSS_FUNC = ""
      option.param.IS_TRANSFER = 0
      option.param.BASE_AI_CD = null
      option.param.NETWORK_NAME = autoModel.NETWORK_NAME
      option.param.NETWORK_PATH = autoModel.NETWORK_PATH
      option.param.MAX_TRIAL = aiInfo.CFG_TRIAL
    }

    if (isEarlyStop) {
      option.param.EARLY_MONITOR = aiInfo.CFG_EARLY_MONITOR
      option.param.EARLY_MODE = aiInfo.CFG_EARLY_MODE
    }

    option.queryId = "setTrainModelInfo"
    await DH.executeQuery(option)

    res.json({ status: 1 })
  })
)

router.post("/getUsableGpu", async (req, res, next) => {
  try {
    let result = await CF.sendRequestResLong(
      config.masterIp,
      config.masterPort,
      C.URI.getUsableGpu,
      {}
    )
    //바이너리 에러처리 추가 되어야 함
    res.json({ status: 1, value: result })
  } catch (error) {
    res.json({ status: 1, value: [] })
    return
  }
})

router.post(
  "/updateAiPrjDataSet",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "updateAiPrj"
    option.param = req.body

    await DH.executeQuery(option)

    option.queryId = "removeTrainDataSet"
    await DH.executeQuery(option)

    option.queryId = "setTrainDataset"
    await DH.executeQuery(option)

    res.json({ status: 1 })
  })
)

router.post("/gettrainmodelinfo", async (req, res, next) => {
  const param = req.body
  let data = await _exeQuery(C.Q.TABAIPRJ, param, "getTrainModelInfo")
  res.json(data)
})

router.post(
  "/getAiSetUpData",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "getTrainDataSetList"
    option.param = req.body

    let datsetList = await DH.executeQuery(option)
    option.param.DATA = []
    datsetList.map((ele) => {
      option.param.DATA.push(ele.DATASET_CD)
    })
    option.queryId = "getDistinctTrainDataSetList"

    let trainDataList = await DH.executeQuery(option)

    let classDataList
    //////태그 정보 호출
    option.queryId = "getTrainClassList"
    let classList = await DH.executeQuery(option)
    if (classList.length <= 0) {
      //클래스가 생성 안되어 있으면 데이터셋_CD에 연결된 모든 클래스 가져오기
      option.queryId = "getTagListByDataSetCD"
      classDataList = await DH.executeQuery(option)

      //드래거블의 스테이트값 설정부분
      classDataList.map((ele) => {
        ele.text = ele.NAME
        ele.state = false
      })
    } else {
      let tagList = []
      classList.map((ele) => {
        let temp = ele.ELEMENT_TAGS.split(",")
        tagList = tagList.concat(temp)
      })
      option.param.DATA = tagList
      option.queryId = "getTagListByTagCD"
      classDataList = await DH.executeQuery(option)

      //드래거블의 스테이트값 설정부분
      classDataList.map((ele) => {
        ele.text = ele.NAME
        ele.state = false
      })
    }

    res.json({
      DATASET: trainDataList,
      CLASSES: classDataList
    })
  })
)

router.post(
  "/setAiClassInfo",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "removeTrainClasses"
    option.param = req.body

    await DH.executeQuery(option)

    option.param.DATA = []

    req.body.CLASSES.map((ele) => {
      option.param.DATA.push({
        AI_CD: req.body.AI_CD,
        NAME: ele.NAME,
        DESC_TXT: ele.DESC_TXT,
        COLOR: ele.COLOR,
        ELEMENT_TAGS: ele.TAG_CD
      })
    })

    option.queryId = "setTrainClassInfo"
    await DH.executeQuery(option)

    option.queryId = "updateAiPrj"
    await DH.executeQuery(option)

    res.json({ status: 1 })
  })
)

router.post(
  "/getTagListByDatasetCD",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "getTagListByDataSetCD_single"
    option.param = req.body
    let tagInfo = await DH.executeQuery(option)
    res.json(tagInfo)
  })
)

router.post(
  "/stopActiveModel",
  asyncHandler(async (req, res, next) => {
    let processDown = await CF.sendRequestResLong(
      config.masterIp,
      config.masterPort,
      C.URI.killProcess,
      { PID: req.body.PID }
    )
    //바이너리 에러처리 추가 되어야 함
    logger.info(`Stop Model [${req.body.PID}] ${processDown}`)
    if (processDown === "True") res.json({ status: 1 })
    else res.json({ status: 0 })
  })
)

router.post(
  "/getDataSetListByAICD",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "getDataSetListByAICD"
    option.param = req.body
    let dataSets = await DH.executeQuery(option)
    res.json(dataSets)
  })
)

router.post(
  "/getModelDataByAICD",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "getModelDataByAICD"
    option.param = req.body
    let model = await DH.executeQuery(option)
    model = model[0]
    if (model?.IS_AUTO) {
      model.IS_AUTO = model.IS_AUTO === 1
      model.IS_TRANSFER = model.IS_TRANSFER === 1
    }
    res.json(model)
  })
)

router.post(
  "/getClassListByAICD",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "getTrainClassList"
    option.param = req.body
    let classes = await DH.executeQuery(option)

    // classes.map((ele) => {
    //   console.log(ele)
    // })

    const promises = classes.map((ele) => {
      return new Promise((resolve, reject) => {
        option.param.DATA = ele.ELEMENT_TAGS.split(",")
        option.param.TRAIN_CLASS_CD = ele.TAG_CD
        option.param.TRAIN_CLASS = ele.NAME
        option.param.TRAIN_COLOR = ele.COLOR
        option.queryId = "getTagListByTagCD"
        let result = DH.executeQuery(option).catch((err) => {
          reject(err)
        })
        resolve(result)
      })
    })

    await Promise.all(promises).then((data) => {
      res.json(data)
    })
  })
)

router.post("/getActiveTrain", async (req, res, next) => {
  let option = {}
  option.source = C.MAPPER.AIPRJ
  option.queryId = "getActiveTrain"
  option.param = req.body
  let activeTrain = await DH.executeQuery(option)
  activeTrain = activeTrain[0].ACTIVE_TRAIN
  activeTrain = activeTrain === undefined ? 0 : activeTrain

  res.json({ ACTIVE_TRAIN: activeTrain })
})

router.post(
  "/startTrainAi",
  asyncHandler(async (req, res, next) => {
    let OBJECT_TYPE = req.body.AI_CD.substr(0, 1)
    const IS_RESUME = req.body.IS_RESUME
    let START_EPOCH = 0
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.queryId = "getTrainClassList"
    option.param = req.body
    let trainTagList = await DH.executeQuery(option)

    if (IS_RESUME) {
      option.queryId = "getLastEpoch"
      let lastEpoch = await DH.executeQuery(option)
      START_EPOCH = lastEpoch[0].LAST_EPOCH
    }

    // className, dpName => CLASS_NO, CLASS_NAME 으로 수정 smpark
    let classTagMap = {}
    let aiParam = []
    let segmentationParam = {}
    segmentationParam.IMAGE_LIST = []
    segmentationParam.CLASSES = []
    trainTagList.map(async (ele) => {
      segmentationParam.CLASSES.push({
        CLASS_CD: ele.TAG_CD,
        COLOR: ele.COLOR
      })
      aiParam.push({
        COLOR: ele.COLOR,
        CLASS_NO: ele.TAG_CD,
        CLASS_NAME: ele.NAME,
        TAGS: ele.ELEMENT_TAGS.split(","),
        files: []
      })
    })

    aiParam.map((ele) => {
      ele.TAGS.map((item) => {
        classTagMap[item] = {
          COLOR: ele.COLOR,
          CLASS_NO: ele.CLASS_NO,
          CLASS_NAME: ele.CLASS_NAME
        }
      })
    })

    const promises = aiParam.map((ele) => {
      return new Promise((resolve, reject) => {
        option.param.DATA = []
        option.param.DATA = ele.TAGS
        //밑에 쿼리 다시 생각해봐야함
        if (OBJECT_TYPE === "C") option.queryId = "getFileListCLF"
        if (OBJECT_TYPE === "D" || OBJECT_TYPE === "S")
          option.queryId = "getFileList"
        let result = DH.executeQuery(option).catch((err) => {
          reject(err)
        })
        resolve(result)
      })
    })

    await Promise.all(promises).then((data) => {
      data.map((ele) => {
        segmentationParam.IMAGE_LIST.push({ IMAGE_PATH: ele.FILE_PATH })
      })
      data.map((ele, idx) => {
        aiParam[idx].TAGS = ele
      })
    })

    let segParam = []
    aiParam.map((ele) => {
      ele.TAGS.map((item) => {
        if (OBJECT_TYPE === "C") ele.files.push(item.FILE_PATH)
        else if (item.ANNO_DATA !== null && item.ANNO_DATA.length > 0) {
          ele.files.push({ PATH: item.FILE_PATH, LABELS: item.ANNO_DATA })
        }
      })
      ele.TAGS = {}
    })

    let item = []
    if (OBJECT_TYPE === "C") item.push(C.BIN.trainCLF)
    else if (OBJECT_TYPE === "D") item.push(C.BIN.trainDETEC)
    else if (OBJECT_TYPE === "S") item.push(C.BIN.trainSEG)

    item.push({
      CLASS_MAP: classTagMap,
      IMAGE_LIST: aiParam,
      // CLASSES: OBJECT_TYPE === "S" ? segmentationParam.CLASSES : null,
      AI_CD: req.body.AI_CD,
      DATA_TYPE: req.body.DATA_TYPE,
      MDL_PATH: "./testModel",
      IS_AUTO: true,
      // CRN_USR: req.body.USER_ID || "user",
      CRN_USR: "user",
      IS_TRANSFER: false,
      GPU_RATE: 30,
      OBJECT_TYPE: OBJECT_TYPE, // 여긴데 내가 수정해놨어요 smpark
      START_EPOCH: START_EPOCH
    })

    option.queryId = "getTrainModelInfo"
    let trainSetting = await DH.executeQuery(option)
    trainSetting = trainSetting[0]
    trainSetting.IS_AUTO = trainSetting.IS_AUTO === 1
    trainSetting.IS_TRANSFER = trainSetting.IS_TRANSFER === 1
    trainSetting.IS_EARLYSTOP = trainSetting.IS_EARLYSTOP === 1

    item[1].GPU_IDX = JSON.parse(trainSetting.GPU_INDEX)
    item[1].MAX_TRIAL = 10 // 기본값
    item[1].IS_EARLYSTOP = trainSetting.IS_EARLYSTOP
    item[1].EARLY_MONITOR = null
    item[1].EARLY_MODE = null
    item[1].IMG_SIZE = trainSetting.IMG_SIZE
    item[1].IMG_CHANNEL = trainSetting.IMG_CHANNEL
    item[1].NETWORK_PATH = trainSetting.NETWORK_PATH
    item[1].NETWORK_NAME = trainSetting.NETWORK_NAME

    if (trainSetting.IS_AUTO) {
      item[1].EPOCH = trainSetting.EPOCH
      item[1].BATCH_SIZE = trainSetting.BATCH_SIZE
      item[1].MAX_TRIAL = trainSetting.MAX_TRIAL
    } else {
      item[1].IS_AUTO = trainSetting.IS_AUTO
      item[1].IS_TRANSFER = trainSetting.IS_TRANSFER
      item[1].EPOCH = trainSetting.EPOCH
      item[1].BATCH_SIZE = trainSetting.BATCH_SIZE
      item[1].OPTIMIZER = trainSetting.OPTIMIZER
      item[1].ACTIVE_FUNC = trainSetting.ACTIVE_FUNC
      item[1].LOSS_FUNC = trainSetting.LOSS_FUNC
      if (trainSetting.IS_TRANSFER) {
        item[1].BASE_AI_CD = trainSetting.BASE_AI_CD
        item[1].NETWORK_PATH = trainSetting.NETWORK_PATH
        item[1].NETWORK_NAME = trainSetting.NETWORK_NAME
      }
    }

    if (trainSetting.IS_EARLYSTOP) {
      item[1].EARLY_MONITOR = trainSetting.EARLY_MONITOR
      item[1].EARLY_MODE = trainSetting.EARLY_MODE
    }
    // fs.writeFileSync("attr.json", JSON.parse(JSON.stringify(item[1])))

    let pid = await CF.sendRequestResLong(
      config.masterIp,
      config.masterPort,
      C.URI.aiTrain,
      item[1]
    )
    CF.resException(pid, C.URI.aiTrain)
    let sendData = {}
    req.body.AI_STS = "READY"
    sendData.TYPE = "STATE_CHANGE"
    sendData.AI_CD = req.body.AI_CD
    sendData.MSG = req.body
    CF.sendSoc(sendData, "TRAIN")

    logger.debug(`READY Trainer Pid ${pid}`)
    // console.log(JSON.stringify(item[1]))

    option.param.AI_STS = "READY"
    option.param.AI_PID = pid
    option.param.TRAIN_SRT_DTM = "now()"
    option.queryId = "updateAiPrj"
    await DH.executeQuery(option)
    res.json({ status: 1, pid: pid })
  })
)

router.post(
  "/stopTrainAi",
  asyncHandler(async (req, res, next) => {
    // let result = await spawn("kill", ["-15", req.body.AI_PID])
    // //  _runProcessSync(["-9", req.body.AI_PID])
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.param = {}
    option.param.AI_CD = req.body.AI_CD
    option.queryId = "getAiPid"
    let pid = await DH.executeQuery(option)
    pid = pid[0]

    option.param.AI_STS = "STOP"
    // option.param.AI_PID = -1
    // option.param.TRAIN_END_DTM = "now()"
    option.queryId = "updateAiPrj"
    await DH.executeQuery(option)
    res.json({ status: 1 })

    let sendData = {}
    sendData.TYPE = "STATE_CHANGE"
    sendData.AI_CD = req.body.AI_CD
    sendData.MSG = option.param
    CF.sendSoc(sendData, "TRAIN")
    _killTrainner(pid.AI_PID, req.body.AI_CD)
  })
)

const _killTrainner = async (pid, AI_CD) => {
  let option = {}
  option.param = { AI_CD: AI_CD }
  option.source = C.MAPPER.AIPRJ
  option.queryId = "getAiPath"
  let path = await DH.executeQuery(option)
  path = path[0].PATH

  let processDown = await CF.sendRequestResLong(
    config.masterIp,
    config.masterPort,
    C.URI.killProcess,
    { PID: pid }
  )
  logger.debug(`[${AI_CD}] Process Down Y/N ${processDown}`)

  option.param = {}
  option.source = C.MAPPER.AIPRJ
  let stopMsg = ""

  if (processDown === "True") {
    logger.info(`[${AI_CD}] Process Stop`)
    option.param.AI_CD = AI_CD
    option.param.AI_STS = "DONE"
    option.param.AI_PID = -1
    option.param.TRAIN_END_DTM = "now()"
    option.queryId = "updateAiPrj"
  } else {
    logger.error(`[${AI_CD}] Process Stop Fail`)
    option.param.AI_CD = AI_CD
    option.param.AI_STS = "FAIL"
    option.param.LAST_MSG = "Process Kill Fail"
    option.param.TRAIN_END_DTM = "now()"
    option.queryId = "updateAiPrj"
  }
  if (path !== null && path !== "") {
    getSize(path, (err, size) => {
      option.param.AI_SIZE = size
    })
  }

  await DH.executeQuery(option)

  let sendData = {}
  sendData.TYPE = "STATE_CHANGE"
  sendData.AI_CD = AI_CD
  sendData.MSG = option.param
  CF.sendSoc(sendData, "TRAIN")
}

router.post(
  "/getTrainResult",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.param = req.body
    option.queryId = "getTrainResult"
    let result = await DH.executeQuery(option)

    let mdlPath = path.join(config.aiPath, req.body.AI_CD)

    let fileList = await fs.readdirSync(mdlPath)
    let existResult = []
    result.map((item) => {
      let idx = fileList.findIndex((ele) => item.EPOCH === parseInt(ele))
      if (idx >= 0) existResult.push(item)
    })

    option.queryId = "getMdlIdxList"
    let idxList = await DH.executeQuery(option)

    let returnResult = []
    idxList.map((mdlIdx) => {
      let temp = { MDL_IDX: mdlIdx.MDL_IDX }
      temp.EPOCHS = []
      result.map((epoch) => {
        if (epoch.MDL_IDX === temp.MDL_IDX) temp.EPOCHS.push(epoch)
      })
      returnResult.push(temp)
    })

    res.json(returnResult)
  })
)

router.post("/getTestAi", async (req, res, next) => {
  const isReady = req.body.IS_READY === undefined ? false : req.body.IS_READY
  let activeCheck = await CF.checkModel(
    req.body.AI_CD,
    isReady,
    req.body.EPOCH_NO
  )
  let result = {}
  let runPredict = activeCheck.runPrc
  let returnCode = activeCheck.code
  result = returnCode

  try {
    if (runPredict) {
      let OBJECT_TYPE = req.body.DATASET_CD.substr(0, 1)
      let mdl_path = path.join(config.aiPath, req.body.AI_CD)
      mdl_path = path.join(mdl_path, String(req.body.EPOCH_NO).padStart(4, 0))
      let option = {}
      option.source = C.MAPPER.IMG_ANNO
      option.param = req.body
      if (req.body.DATA_CD !== undefined) option.queryId = "getImage"
      else option.queryId = "getImageList"

      option.param.DATA_STATUS = "ORG"
      let dataList = await DH.executeQuery(option)
      let imageInfo = []
      dataList.map((ele) => {
        imageInfo.push({
          IMAGE_PATH: ele.FILE_PATH,
          DATASET_CD: ele.DATASET_CD, //바이페스
          DATA_CD: ele.DATA_CD, //바이페스
          COLOR: null,
          RECT: null,
          START_FRAME: req.body.START_FRAME || null,
          END_FRAME: req.body.END_FRAME || null
        })
      })

      option.source = C.MAPPER.AIPRJ
      option.queryId = "getTrainClassList"
      let classList = await DH.executeQuery(option)

      // EPOCH 안들어오길래 수정했어요 smpark
      let tempStr = {
        IMAGES: imageInfo,
        OBJECT_TYPE: OBJECT_TYPE,
        DATA_TYPE: req.body.DATA_TYPE,
        AI_CD: req.body.AI_CD,
        MDL_PATH: mdl_path,
        EPOCH: req.body.EPOCH_NO,
        IS_TEST: true
      }

      // tempStr로 수정했으요 smpark
      let mdlPid = await CF.sendRequestResLong(
        config.masterIp,
        config.masterPort,
        C.URI.modelLoad,
        tempStr
      )
      if (mdlPid.PID === 0) throw new Error("Model Load Fail")

      // EPOCH 안들어오길래 수정했어요 smpark
      let testResult = await CF.sendRequestResLong(
        config.masterIp,
        config.masterPort,
        C.URI.miniPredictor,
        tempStr
      )
      CF.resException(testResult, C.URI.miniPredictor)

      result = []
      if (OBJECT_TYPE === "C") {
        testResult = testResult[0]
        testResult.ANNO_DATA.map((ele) => {
          let test = []
          test = classList.filter((item) => item.NAME === ele.CLASS_DB_NM)
          result.push({
            IMG_ORG: ele.IMG_PATH,
            CLASS: `${test[0]?.NAME}`,
            POSITION: [],
            ACCURACY: ele.ACCURACY
          })
        })
      } else {
        result = testResult
      }
      logger.info(`Predict Test Done`)
    }
    res.json(result)
  } catch (error) {
    logger.error(`[${req.body.AI_CD}:${req.body.EPOCH}] Model LoadFail`)
    res.json({ status: 0 })
  }
})

router.post(
  "/removeAiModel",
  asyncHandler(async (req, res, next) => {
    try {
      const datasetDir = path.join(config.aiPath, req.body.AI_CD)
      fs.existsSync(datasetDir) && rimraf.sync(datasetDir)

      await CF.exeQuery(C.Q.AIPRJ, req.body, "removeTrainDataSet")
      await CF.exeQuery(C.Q.AIPRJ, req.body, "removeTrainClassInfo")
      await CF.exeQuery(C.Q.TABAIPRJ, req.body, "removeTrainLog")
      await CF.exeQuery(C.Q.TABAIPRJ, req.body, "removeSelectedModel")
      await CF.exeQuery(C.Q.TABAIPRJ, req.body, "removetrainModel")
      await CF.exeQuery(C.Q.TABAIPRJ, req.body, "removeModelSummary")

      await CF.exeQuery(C.Q.AIPRJ, req.body, "removeAi")

      res.json({ status: 1 })
    } catch (error) {
      res.json({ status: 0 })
    }
  })
)

router.post(
  "/getDownloadAi",
  asyncHandler(async (req, res, next) => {
    let mdl_path = path.join(config.aiPath, req.body.AI_CD)
    let ai_path = path.join(mdl_path, String(req.body.EPOCH_NO).padStart(4, 0))
    let fileName = `${req.body.AI_CD}_${String(req.body.EPOCH_NO).padStart(
      4,
      0
    )}.zip`
    const archive = archiver("zip", { zlib: { level: 9 } })
    const output = fs.createWriteStream(`${mdl_path}/${fileName}`)

    logger.info(`Model Achive [${mdl_path}/${fileName}} Create`)

    if (fs.existsSync(`${mdl_path}/${fileName}`)) {
      logger.info(`Model Achive Create Success [${mdl_path}/${fileName}}`)
      const mimetype = mime.getType(`${mdl_path}/${fileName}`)
      res.setHeader("Content-disposition", "attachment; filename=" + fileName)
      res.setHeader("Content-type", mimetype)

      res.download(`${mdl_path}/${fileName}`)
    } else {
      logger.info(`Model Achive Start [${mdl_path}/${fileName}}`)
      new Promise((resolve, reject) => {
        output.on("close", () => {
          console.log(archive.pointer() + " total bytes")
          resolve(output)
        })

        output.on("end", () => {
          console.log("Data has been drained")
        })

        // good practice to catch warnings (ie stat failures and other non-blocking errors)
        archive.on("warning", (err) => {
          console.log("warning")
          reject(err)
        })

        archive.on("error", (err) => {
          console.log("error")
          reject(err)
        })

        archive.pipe(output)
        archive.file(`${mdl_path}/classes.names`, { name: "classes.names" })
        archive.file(`${mdl_path}/classes.json`, { name: "classes.json" })
        archive.directory(ai_path, "model")

        archive.finalize()
      })
        .then((resolve) => {
          logger.info(`Model Achive Create Success [${mdl_path}/${fileName}}`)
          const mimetype = mime.getType(`${mdl_path}/${fileName}`)
          res.setHeader(
            "Content-disposition",
            "attachment; filename=" + fileName
          )
          res.setHeader("Content-type", mimetype)

          res.download(`${mdl_path}/${fileName}`)
        })
        .catch((reject) => console.log(reject))
    }
  })
)

router.post(
  "/getExpertCode",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.param = {}
    option.param.CODE_LOCATION = "NEW_AI"
    option.queryId = "getCodeInfoByLocation"
    const codeList = await DH.executeQuery(option)
    res.json(codeList)
  })
)

router.post(
  "/getExpertBaseModels",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.param = req.body
    option.queryId = "getBaseModelByObjectType"
    const baseList = await DH.executeQuery(option)
    res.json(baseList)
  })
)

router.post(
  "/getExpertMyModels",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = C.MAPPER.AIPRJ
    option.param = req.body
    option.queryId = "getMyModelByObjectType"
    let baseList = await DH.executeQuery(option)

    option.param.DATA = baseList
    option.queryId = "getMyModelClassList"
    let classList = []

    if (baseList.length > 0) classList = await DH.executeQuery(option)

    baseList.map((ele) => {
      ele.CLASSES = classList.filter((item) => item.AI_CD === ele.AI_CD)
    })
    console.log(baseList)
    res.json(baseList)
  })
)

function _runProcess(args) {
  const ls = spawn("python", args)
  return ls.pid
}
function _runProcessSync(args) {
  return new Promise((resolve, reject) => {
    const ls = spawn("python", args)
    let result = ""
    let Err = ""
    ls.stdout.on("data", function (data) {
      result += data
      console.log(String(data))
    })

    ls.stderr.on("data", function (data) {
      Err += data
      console.log("ERROR  " + String(data))
    })
    ls.on("exit", function (code) {
      if (code === 0) {
        resolve(String(result))
      } else {
        console.log(String(Err))
        reject(String(Err))
      }
    })
  })
}

function _runZipync(args) {
  console.log(args)
  return new Promise((resolve, reject) => {
    const ls = spawn("zip", args)
    let result = ""
    let Err = ""
    ls.stdout.on("data", function (data) {
      result += data
      console.log(String(data))
    })

    ls.stderr.on("data", function (data) {
      Err += data
      console.log("ERROR  " + String(data))
    })
    ls.on("exit", function (code) {
      if (code === 0) {
        resolve(String(result))
      } else {
        console.log(String(Err))
        reject(String(Err))
      }
    })
  })
}

router.post(
  "/updateAiInfo",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.param = {}
    option.source = C.MAPPER.AIPRJ
    option.param.TITLE = req.body.TITLE
    option.param.DESC_TXT = req.body.DESC_TXT
    option.param.GPU_LIMIT = req.body.CFG_GPU
    option.param.EPOCH = req.body.CFG_EPOCH
    option.param.BATCH_SIZE = req.body.CFG_BATCH
    option.param.IS_AUTO = req.body.CFG_MODE === "Auto" ? true : false
    option.param.GPU_INDEX = JSON.stringify(req.body.CFG_GPUIDX)
    option.param.AI_CD = req.body.AI_CD
    option.queryId = "updateAiPrj"
    await DH.executeQuery(option)

    option.queryId = "updateTrainModelInfo"
    await DH.executeQuery(option)
    res.json({ status: 1 })
  })
)

router.post("/getselectedModel", async (req, res, next) => {
  let param = {}
  let list = await _exeQuery(C.Q.AIPRJ, param, "getSelectedModel")

  //이미지, 비디오, 리얼타임의 경우 디폴트 모델 제공
  if (
    req.body.DATA_TYPE === "I" ||
    req.body.DATA_TYPE === "V" ||
    req.body.DATA_TYPE === "R"
  ) {
    let baseList = await _exeQuery(C.Q.IS, param, "getBaseModelDistinct")
    res.json([...baseList, ...list])
  } else {
    res.json(list)
  }
})

router.post("/gettrainedmodel", async (req, res, next) => {
  let option = {}
  option.param = {}
  option.source = C.MAPPER.AIPRJ
  option.param.AI_CD = req.body.AI_CD
  option.param.DATA_TYPE = req.body.DATA_TYPE
  option.queryId = "getTrainedModelList"
  let list = await DH.executeQuery(option)
  res.json(list)
})

router.post("/settrainedmodel", async (req, res, next) => {
  let option = {}
  option.param = {}
  option.source = C.MAPPER.AIPRJ
  option.param.AI_CD = req.body.AI_CD
  option.param.DATA_TYPE = req.body.DATA_TYPE
  option.param.OBJECT_TYPE = req.body.OBJECT_TYPE
  option.param.EPOCH = req.body.EPOCH
  option.param.MDL_IDX = req.body.MDL_IDX

  option.queryId = "removeSelectedModel"
  await DH.executeQuery(option)

  option.queryId = "setTrainedModel"
  await DH.executeQuery(option)
  res.json({ status: 1 })
})

router.post("/removeSelectedModel", async (req, res, next) => {
  let option = {}
  option.param = {}
  option.source = C.MAPPER.AIPRJ
  option.param.AI_CD = req.body.AI_CD

  option.queryId = "removeSelectedModel"
  await DH.executeQuery(option)

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
