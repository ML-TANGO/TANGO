import express from "express"
import asyncHandler from "express-async-handler"
import fs from "fs"
import moment from "moment"
import nodemailer from "nodemailer"
import smtpTransport from "nodemailer-smtp-transport"
import getSize from "get-folder-size"

import path from "path"
import CC from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"
import DH from "../lib/DatabaseHandler"

import { createLogger } from "winston"
const logger = require("../lib/Logger")(__filename)

const router = express.Router()
const config = CL.getConfig()
const spawn = require("child_process").spawn
const CRN_USR = "testUser@test.co.kr"

router.post(
  "/binErrorHandle",
  asyncHandler(async (req, res, next) => {
    // console.log("==================================!!!!==============")
    // console.log(JSON.stringify(req.body, null, 1))
    let data = req.body.MESSAGE

    switch (data.PRC_NAME) {
      case "Trainer2.py":
        try {
          data.MSG = JSON.parse(data.MSG)
        } catch (error) {}

        if (data.MSG.ERR_TYPE === "OOM") {
          let trainReq = data.MSG
          logger.info(
            `Trainner OUT OF MEMORY [ ${data.PID} ] [${trainReq.AI_CD}] In BatchSize[${trainReq.BATCH_SIZE}]`
          )
          if (trainReq.BATCH_SIZE === 1) {
            data.AI_STS = "FAIL"
            data.AI_MSG = "Out Of Memory"
            _trainnerDown(data)
          } else _reTrain(trainReq, data.PID)
        } else {
          logger.info(`Trainner Process Killed [ ${data.PID} ] ${data.MSG}`)
          data.AI_STS = "FAIL"
          _trainnerDown(data)
        }
        break
      case "Worker2.py":
        try {
          data = JSON.parse(data.MSG)
          _workerDown(data.IMAGES[0].DATASET_CD, data.ERR)
        } catch (error) {}

        break
      default:
        break
    }

    res.json({ status: 1 })
  })
)

const _workerDown = async (DATASET_CD, ERR) => {
  let option = {}
  option.source = CC.MAPPER.DATASET
  option.param = {}
  option.param.DATASET_CD = DATASET_CD
  option.param.DATASET_STS = "AUTO_FAIL"
  option.param.LAST_MSG = ERR
  option.param.PID = -1
  option.queryId = "setUpdateDataset"
  await DH.executeQuery(option)

  logger.error(`Predictor Process Fail [ ${DATASET_CD} ] \n${ERR}`)
}

const _reTrain = async (trainReq, pid) => {
  let batchSize = trainReq.BATCH_SIZE / 2
  logger.info(`[${trainReq.AI_CD}] ReTrain Ready`)
  //check ProcessDown

  let processDown = await CF.sendRequestResLong(
    config.masterIp,
    config.masterPort,
    CC.URI.killProcess,
    { PID: pid }
  )
  logger.debug(`[${trainReq.AI_CD}] Process Down Y/N ${processDown}`)
  if (processDown === "True") {
    let option = {}
    option.source = CC.MAPPER.AIPRJ
    option.param = {}
    option.param.AI_CD = trainReq.AI_CD
    option.param.AI_STS = "STOP"
    option.queryId = "updateAiPrj"
    await DH.executeQuery(option)

    _sendTrainSocData(
      "STATE_CHANGE",
      trainReq.AI_CD,
      option.param,
      option.param.AI_STS
    )

    logger.info(
      `[${trainReq.AI_CD}] ReTrain Reduce Batch_Size [${trainReq.BATCH_SIZE} -> ${batchSize}] Start`
    )
    trainReq.BATCH_SIZE = batchSize
    let pid = await CF.sendRequestResLong(
      config.masterIp,
      config.masterPort,
      CC.URI.aiTrain,
      trainReq
    )
    log.debug(`============ RETRAIN PID [ ${pid} ]`)
    CF.resException(pid, CC.URI.aiTrain)
  } else {
    //프로세스 다운 실패.
  }
}

const _trainnerDown = async (data) => {
  let option = {}
  option.source = CC.MAPPER.BIN
  option.param = {}
  option.param.AI_PID = data.PID
  option.queryId = "getAIfromPid"
  let ai_list = await DH.executeQuery(option)
  if (ai_list.length > 0) {
    option.param.DATA = ai_list
    option.param.TRAIN_END_DTM = "now()"
    option.param.AI_STS = data.AI_STS
    option.param.AI_PID = -1
    option.param.LAST_MSG = data.MSG
    option.queryId = "stopAiArray"
    await DH.executeQuery(option)

    ai_list.map((ele) => {
      _sendTrainSocData("STATE_CHANGE", ele.AI_CD, data, data.AI_STS)
    })
  }
}

const _sendTrainSocData = (type, AI_CD, MSG, AI_STS) => {
  let sendData = {}
  sendData.TYPE = type
  sendData.AI_CD = AI_CD
  sendData.MSG = MSG
  sendData.AI_STS = AI_STS
  CF.sendSoc(sendData, "TRAIN")
}

router.post(
  "/binTestGetClass",
  asyncHandler(async (req, res, next) => {
    const jsonFile = req.body.DAT_PATH

    const datFile = JSON.parse(fs.readFileSync(jsonFile))

    let tempCls = []
    datFile.map((ele) => {
      tempCls.push(ele.label)
    })

    tempCls = new Set(tempCls)
    console.log(tempCls)
    res.json(tempCls)
  })
)

router.post(
  "/binTest_setDetactionDat",
  asyncHandler(async (req, res, next) => {
    const datasetDir = "/Users/upload/DataSets/DI200316/"
    const jsonFile =
      "/Users/dmshin/Weda/B0002/dogs-vs-cats/TEST_8000/datfile/3번 양품/data.json"

    const datFile = JSON.parse(fs.readFileSync(jsonFile))
    console.log(datasetDir)
    let fileList = fs.readdirSync(datasetDir)

    // let temp = datFile.filter(item => item.path === "dataset/1_1_1_0_1599544735738.jpg" )

    fileList.map((ele, idx) => {
      if (path.parse(ele).ext === ".jpg" && !ele.includes("번불랴")) {
        let dat = path.parse(ele).name + ".dat"
        dat = path.join(datasetDir, dat)

        let classes = datFile.filter((item) => item.path === "dataset/" + ele)

        let datBase = { POLYGON_DATA: [] }

        classes.map((oldDatFile) => {
          let clsabnormal = {
            DATASET_CD: "DI200306",
            DATA_CD: "00000000",
            TAG_CD: 1533,
            TAG_NAME: "abnormal",
            CLASS_CD: null,
            COLOR: "#ff0004",
            CURSOR: "isRect",
            NEEDCOUNT: 2,
            POSITION: []
          }

          let class3 = {
            DATASET_CD: "DI200306",
            DATA_CD: "00000000",
            TAG_CD: 1534,
            TAG_NAME: "3",
            CLASS_CD: null,
            COLOR: "#ffc300",
            CURSOR: "isRect",
            NEEDCOUNT: 2,
            POSITION: []
          }

          let class0 = {
            DATASET_CD: "DI200306",
            DATA_CD: "00000000",
            TAG_CD: 1535,
            TAG_NAME: "0",
            CLASS_CD: null,
            COLOR: "#00ffe4",
            CURSOR: "isRect",
            NEEDCOUNT: 2,
            POSITION: []
          }

          let class2 = {
            DATASET_CD: "DI200306",
            DATA_CD: "00000000",
            TAG_CD: 1536,
            TAG_NAME: "2",
            CLASS_CD: null,
            COLOR: "#009bff",
            CURSOR: "isRect",
            NEEDCOUNT: 2,
            POSITION: []
          }
          let class230 = {
            DATASET_CD: "DI200306",
            DATA_CD: "00000000",
            TAG_CD: 1537,
            TAG_NAME: "230",
            CLASS_CD: null,
            COLOR: "#dc00ff",
            CURSOR: "isRect",
            NEEDCOUNT: 2,
            POSITION: []
          }

          let class30 = {
            DATASET_CD: "DI200306",
            DATA_CD: "00000000",
            TAG_CD: 1538,
            TAG_NAME: "30",
            CLASS_CD: null,
            COLOR: "#00ff1d",
            CURSOR: "isRect",
            NEEDCOUNT: 2,
            POSITION: []
          }

          let classnormal = {
            DATASET_CD: "DI200316",
            DATA_CD: "00000000",
            TAG_CD: 1559,
            TAG_NAME: "normal",
            CLASS_CD: null,
            COLOR: "#00ff18",
            CURSOR: "isRect",
            NEEDCOUNT: 2,
            POSITION: []
          }

          let temp = {}
          if (oldDatFile.label === "0") temp = class0
          else if (oldDatFile.label === "3") temp = class3
          else if (oldDatFile.label === "2") temp = class2
          else if (oldDatFile.label === "230") temp = class230
          else if (oldDatFile.label === "30") temp = class30
          else if (oldDatFile.label === "abnormal") temp = clsabnormal
          else if (oldDatFile.label === "normal") temp = classnormal

          // console.log(oldDatFile)
          temp.POSITION.push({
            X: parseInt(oldDatFile.x),
            Y: parseInt(oldDatFile.y)
          })
          temp.POSITION.push({
            X: parseInt(oldDatFile.x) + parseInt(oldDatFile.w),
            Y: parseInt(oldDatFile.y) + parseInt(oldDatFile.h)
          })
          datBase.POLYGON_DATA.push(temp)
        })
        // console.log(JSON.stringify(datBase, null, 1))
        fs.writeFileSync(dat, JSON.stringify(datBase), {
          encoding: "utf8",
          flag: "w"
        })
        console.log(`DONE ${dat}  (${idx}/${fileList.length})`)
      }
    })

    console.log("============Done=====================")
    res.json({ status: "1" })
  })
)

router.post("/staticPredict", async (req, res, next) => {
  // req.body =
  //   '{"RESULT":[{"FILE_PATH":"/Users/upload/InputSources/86/data/FudanPed00070.dat","IMAGE_PATH":"/Users/upload/InputSources/86/data/FudanPed00070.png","DATASET_CD":86,"DATA_CD":222,"TAGS":[{"CLASS_DB_NM":"person","COLOR":"#406a9e","ACCURACY":0.9963463544845581}],"BASE_MDL":86,"TOTAL_FRAME":0,"FPS":0},{"FILE_PATH":"/Users/upload/InputSources/86/data/FudanPed00069.dat","IMAGE_PATH":"/Users/upload/InputSources/86/data/FudanPed00069.png","DATASET_CD":86,"DATA_CD":223,"TAGS":[{"CLASS_DB_NM":"person","COLOR":"#406a9e","ACCURACY":0.9995943307876587}],"BASE_MDL":86,"TOTAL_FRAME":0,"FPS":0}]}'
  // req.body = JSON.parse(req.body)
  // console.log(JSON.stringify(req.body))

  const data = req.body.RESULT
  let is_cd = data[0].DATASET_CD

  try {
    let objectType = ""
    let isType = ""
    let option = {}

    option.source = CC.MAPPER.IS
    option.param = {}
    option.param.IS_CD = is_cd
    option.queryId = "getOutputType"
    const outputList = await DH.executeQuery(option)

    objectType = outputList[0].OBJECT_TYPE
    isType = outputList[0].IS_TYPE
    let foundClasses = []
    data.map((ele) => {
      ele.TAGS.map((tag) => {
        let savedTag = outputList.find(
          (item) => item.CLASS_NAME === tag.CLASS_DB_NM
        )
        if (savedTag !== undefined) {
          let result_url = "/qithum/"
          result_url += is_cd + "/data"
          result_url = path.join(result_url, path.basename(ele.FILE_PATH))

          let preSavedTag = foundClasses.findIndex(
            (item) => item.CLASS_CD === savedTag.CLASS_CD
          )

          if (preSavedTag === -1)
            foundClasses.push({
              FILE_SEQ: ele.DATA_CD,
              CLASS_CD: savedTag.CLASS_CD,
              COLOR: savedTag.COLOR,
              DP_LABEL: savedTag.DP_LABEL,
              LOCATION: "S",
              ACCURACY: tag.ACCURACY,
              RESULT_PATH: ele.FILE_PATH,
              OUTPUT_PATH: ele.IMAGE_PATH,
              RESULT_URL: result_url,
              OBJECT_TYPE: objectType,
              IS_TYPE: isType,
              FI: null,
              CLASS_CNT: 1
            })
          else foundClasses[preSavedTag].CLASS_CNT++
        }
      })
    })
    option.source = CC.MAPPER.RPT
    option.param.FOUND_CLASSES = foundClasses
    option.queryId = "setStaticReportData"
    if (foundClasses.length > 0) await DH.executeQuery(option)

    option.param.DATA = data
    option.queryId = "updateFileStatus"
    await DH.executeQuery(option)

    option.source = CC.MAPPER.QP
    option.queryId = "updateIsSts"
    option.param.IS_CD = is_cd
    option.param.HW_PID = null
    option.param.SRV_PID = null
    option.param.IS_STS = "DONE"
    option.param.STATUS_MSG = ""
    await DH.executeQuery(option)

    logger.info(
      `IS_CD [${is_cd}] Static Predict Success Predicted [${data.length}]files.`
    )
  } catch (error) {
    logger.error(`IS_CD [${is_cd}] Static Predict Faill !!\n${error.stack}`)
  }

  res.json({ status: 1 })
})

router.post(
  "/prePredictVideo",
  asyncHandler(async (req, res, next) => {
    // req.body = ""
    // req.body = JSON.parse(req.body)
    // console.log("==================BIN======================")

    // console.log(JSON.stringify(req.body))
    // logger.set("app", "i", "AutoLabeling Done \n" + JSON.stringify(req.body))
    // console.log("===========================================")

    //1. 데이터셋 상태 변경
    //2. 프리트레인트 클래스 가져오기
    //3. 태그 저장
    //4. 데이터 엘리먼트 수정
    const param = req.body.RESULT
    const DATASET_CD = param[0].DATASET_CD
    const DATA_CD = param[0].DATA_CD
    const BASE_MDL = param[0].BASE_MDL

    //데이터셋 상태 변경
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.param = {}
    option.param.DATASET_CD = DATASET_CD
    // option.param.DATASET_STS = "DONE"
    // option.queryId = "updateDataSetStatus"
    // await DH.executeQuery(option)

    option.source = CC.MAPPER.BIN
    option.queryId = "initDataTags"
    await DH.executeQuery(option)

    option.param.DATA = []

    let tagDatas = []
    param.map((ele) => {
      tagDatas = tagDatas.concat(ele.TAGS)
      option.param.DATA.push({
        DATA_CD: ele.DATA_CD,
        ANNO_DATA: ele.FILE_PATH
      })
    })

    //데이터 엘리먼트 수정
    option.source = CC.MAPPER.BIN
    option.param.IS_ANNO = true

    option.queryId = "setUpdateDataElement"
    await DH.executeQuery(option)

    //여기부터 클래스 맵핑
    const classDBNames = tagDatas
      .map((ele) => ele.CLASS_DB_NM)
      .filter((v, i, s) => s.indexOf(v) === i)
    tagDatas = tagDatas
      .map((ele) => JSON.stringify(ele))
      .filter((v, i, s) => s.indexOf(v) === i)
    tagDatas = tagDatas.map((ele) => JSON.parse(ele))

    // pre train model list 가져와서 현재 BASE_MDL이 pre train model 인지 확인
    option.queryId = "getPrdTrainModelList"
    let preTrainList = await DH.executeQuery(option)
    const isBaseMdl = preTrainList.some((ele) => ele.BASE_MDL === BASE_MDL)

    //프리트레인드 클래스 가져오기
    option.param.DATA = classDBNames

    let tmpDataType = param[0].DATASET_CD.substr(0, 1)
    option.param.BASE_MDL = BASE_MDL
    let tagList = []
    if (option.param.DATA.length > 0 && isBaseMdl) {
      // 프리트레인드 클래스 가져오기
      option.queryId = "getPrePredictTag"
      tagList = await DH.executeQuery(option)
    } else if (option.param.DATA.length > 0 && !isBaseMdl) {
      // ai model class 가져오기
      option.queryId = "getAiModelTag"
      tagList = await DH.executeQuery(option)
    }

    option.param.DATA = []

    tagList.map((ele) => {
      let temp = tagDatas.find((item) => item.CLASS_DB_NM === ele.CLASS_DB_NAME)
      option.param.DATA.push({
        NAME: ele.CLASS_DP_NAME,
        CLASS_CD: ele.CLASS_CD,
        COLOR: temp.COLOR
      })
    })
    //데이터 테크 설정
    if (option.param.DATA.length !== 0) {
      option.queryId = "setDataTags"
      await DH.executeQuery(option)
    }

    if (isBaseMdl) {
      //pre train model
      option.queryId = "getDataTags"
    } else {
      // ai model
      option.queryId = "getDataTagsByAiModel"
    }
    let savedTag = await DH.executeQuery(option)
    let tagInfo = {}
    let tagCnt = 0
    savedTag.map((ele) => {
      if (ele.TAG_CD !== undefined) {
        try {
          tagInfo[ele.COLOR] = {}
          tagInfo[ele.COLOR].TAG_CD = ele.TAG_CD
          tagInfo[ele.COLOR].CLASS_DP_NAME = ele.CLASS_DP_NAME
          tagCnt++
        } catch (error) {
          console.log("============error=========")
          console.log(error)
        }
      }
    })

    logger.debug(
      `[${DATASET_CD}] AUTOLABEL PreProcessing Tag Info \n${JSON.stringify(
        tagInfo,
        null,
        1
      )}`
    )
    //결과 파일 수정
    option.param.DATA = []
    let annoCtn = 0
    const chFile = param.map((ele) => {
      return new Promise((resolve, reject) => {
        let uniqTagCds = []
        var rsFile = JSON.parse(String(fs.readFileSync(ele.FILE_PATH)))
        rsFile.POLYGON_DATA.map((frame) => {
          if (frame.length > 0) {
            frame.map((obj) => {
              if (tagInfo[obj.COLOR] !== undefined) {
                try {
                  obj.DATASET_CD = DATASET_CD
                  obj.DATA_CD = DATA_CD
                  obj.TAG_CD = tagInfo[obj.COLOR].TAG_CD
                  obj.TAG_NAME = tagInfo[obj.COLOR].CLASS_DP_NAME
                  annoCtn++
                } catch (error) {
                  logger.error(
                    `[${DATASET_CD}][${DATA_CD}] Tag Write Error [${error.stack}]`
                  )
                }
              }
            })
          }
        })
        option.param.DATA.push({
          DATA_CD: ele.DATA_CD,
          TAG_CNT: tagCnt,
          ANNO_CNT: annoCtn
        })
        fs.writeFileSync(ele.FILE_PATH, JSON.stringify(rsFile), {
          encoding: "utf8",
          flag: "w"
        })
        resolve(1)
      })
    })

    await Promise.all(chFile)

    option.source = CC.MAPPER.IMG_ANNO
    option.queryId = "setUpdateDataElementArray"
    await DH.executeQuery(option)

    // _sendEmail(
    //   "dmshin@weda.kr",
    //   "[BluAi] 예측 완료",
    //   "요청하신 데이터셋의 사전 예측이 완료되었습니다." +
    //     JSON.stringify(req.body)
    // )
    res.json({ status: 1 })

    option.param = {}
    option.param.DATASET_CD = DATASET_CD
    option.param.DATASET_STS = "DONE"
    option.param.PID = -1
    option.queryId = "updateDataSetStatus"
    await DH.executeQuery(option)
  })
)

router.post(
  "/test",
  asyncHandler(async (req, res, next) => {
    const now = moment()
    const time = now.format("YYYY-MM-DD HH:mm:ss")
    console.log(time)
  })
)

router.post(
  "/setBinaryException",
  asyncHandler(async (req, res, next) => {
    console.log("===========Error=============")
    console.log(req.body)
  })
)

router.post(
  "/trainBinLog",
  asyncHandler(async (req, res, next) => {
    try {
      logger.debug(`[TrainBinLog] JSON\n${JSON.stringify(req.body, null, 1)}`)
    } catch (error) {
      logger.debug(`[TrainBinLog] STRING\n${req.body}`)
    }

    if (req.body.STATUS === 0) req.body = JSON.parse(req.body.MESSAGE)

    const binLogType =
      req.body.TYPE === undefined ? "EPOCH_CHANGE" : req.body.TYPE
    let option = {}

    option.param = { AI_CD: req.body.AI_CD }
    option.source = CC.MAPPER.AIPRJ
    option.queryId = "getAiPath"
    let path = await DH.executeQuery(option)
    path = path[0].PATH

    option.param = {}
    if (path !== null && path !== "") {
      getSize(path, (err, size) => {
        option.param.AI_SIZE = size
      })
    }

    if (binLogType === "LOG")
      logger.debug(
        `[TrainBinLog Recevie Process Message]\n${JSON.stringify(
          req.body,
          null,
          1
        )}`
      )
    else if (binLogType === "STATE_DOWN") {
      logger.info(`[TrainBinLog] Process Exit ( ${req.body.PID} )`)
      _trainnerDown(req.body)
    } else if (binLogType === "STATE_CH") {
      logger.info(
        `[TrainBinLog] AI Project State Change ( ${req.body.PID} ) [${req.body.AI_CD}]`
      )
      option.source = CC.MAPPER.AIPRJ
      option.queryId = "updateAiPrj"

      option.param.AI_PID = req.body.PID
      option.param.AI_STS = req.body.AI_STS
      option.param.AI_CD = req.body.AI_CD
      await DH.executeQuery(option)

      _sendTrainSocData(
        "STATE_CHANGE",
        req.body.AI_CD,
        req.body,
        req.body.AI_STS
      )
    } else {
      //EPOCH_CHANGE
      if (req.body.IS_LAST) {
        // logger.set("app", "i", "[BIN] Trainer end \n", "end")
        option.source = CC.MAPPER.AIPRJ
        option.param.AI_STS = "DONE"
        option.param.AI_CD = req.body.AI_CD
        option.param.TRAIN_END_DTM = "now()"
        option.queryId = "updateAiPrj"
        await DH.executeQuery(option)
        // _sendEmail(
        //   "dmshin@weda.kr",
        //   "[BluAi] 학습 완료",
        //   "요청하신 AI의 학습이 완료되었습니다." + JSON.stringify(req.body)
        // )
      } else if (req.body.AI_CD !== undefined) {
        //이포크 등록
        option.source = CC.MAPPER.BIN

        if (req.body.EPOCH === 1) {
          logger.debug(
            `[TrainBinLog] !!! Init Train Log By JSON\n${JSON.stringify(
              req.body,
              null,
              1
            )}`
          )
          option.queryId = "initEpoch"
          option.param = req.body
          await DH.executeQuery(option)
        }

        option.queryId = "setTrainLog"
        option.param = req.body
        await DH.executeQuery(option)
        req.body.UPT_DTM = moment().format("YYYY-MM-DD HH:mm:ss")
        req.body.MDL_IDX = 0
        _sendTrainSocData("INFO_UPDATE", req.body.AI_CD, req.body)
      } else
        logger.error(
          `[TrainBilLog] unexpected Trainner Error \n${JSON.stringify(
            req.body,
            null,
            1
          )}`
        )
    }
    res.json({ status: 1 })
  })
)

router.post("/predictBinLog", async (req, res, next) => {
  logger.debug("===============predictBinLog=================")
  try {
    logger.debug(`[PredictBinLog] JSON\n${JSON.stringify(req.body, null, 1)}`)
  } catch (error) {
    logger.debug(`[PredictBinLog] STRING\n${req.body}`)
  }

  if (req.body.STATUS === 0) req.body = JSON.parse(req.body.MESSAGE)

  res.json({ status: 1 })
})
async function _sendEmail(RECEIVER, SUBJECT, TEXT) {
  return 1
  const smtp = config.email
  var transporter = nodemailer.createTransport(
    smtpTransport({
      service: "gmail",
      host: "smtp.gmail.com",
      auth: {
        user: "info.bluai@gmail.com",
        pass: "weda0717!"
      },
      // host: smtp.host,
      // port: smtp.port,
      secure: false,
      requireTLS: false
    })
  )

  var mailOptions = {
    from: "info.bluai@gmail.com",
    to: RECEIVER,
    subject: SUBJECT,
    text: TEXT
  }

  transporter.sendMail(mailOptions, function (error, info) {
    if (error) {
      console.log(error)
      // logger.set("app", "e", "[BIN] Email Send Fail")
      // logger.set("app", "e", String(error))
      res.status(500).send({
        error: "Email Send Error",
        msg: String(error.message)
      })
    } else {
      res.json({
        status: true
      })
    }
  })
}

function _getRandomColor() {
  var letters = "0123456789ABCDEF"
  var color = "#"
  for (var i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)]
  }
  return color
}
router.post(
  "/prePredict",
  asyncHandler(async (req, res, next) => {
    // console.log("==============================================")
    // console.log("==============================================")

    // console.log("==============================================")
    // //test
    // req.body = `{"RESULT":[{"FILE_PATH":"/Users/upload/DataSets/DI210056/1_2_108_107_1599546971516.dat","IMAGE_PATH":"/Users/upload/DataSets/DI210056/1_2_108_107_1599546971516.jpg","DATASET_CD":"DI210056","DATA_CD":"00000000","TAGS":[{"CLASS_DB_NM":"1535","COLOR":"#7499EA","ACCURACY":0.8480584025382996},{"CLASS_DB_NM":"1534","COLOR":"#964964","ACCURACY":0.8414896726608276},{"CLASS_DB_NM":"1536","COLOR":"#3D12CA","ACCURACY":0.8111828565597534},{"CLASS_DB_NM":"1533","COLOR":"#8D3044","ACCURACY":0.6754457950592041}],"BASE_MDL":"DI20210061","TOTAL_FRAME":0,"FPS":0},{"FILE_PATH":"/Users/upload/DataSets/DI210056/1_2_109_108_1599546971516.dat","IMAGE_PATH":"/Users/upload/DataSets/DI210056/1_2_109_108_1599546971516.jpg","DATASET_CD":"DI210056","DATA_CD":"00000001","TAGS":[{"CLASS_DB_NM":"1534","COLOR":"#964964","ACCURACY":0.6766821146011353},{"CLASS_DB_NM":"1535","COLOR":"#7499EA","ACCURACY":0.5957455635070801},{"CLASS_DB_NM":"1533","COLOR":"#8D3044","ACCURACY":0.5304142832756042}],"BASE_MDL":"DI20210061","TOTAL_FRAME":0,"FPS":0},{"FILE_PATH":"/Users/upload/DataSets/DI210056/1_2_110_109_1599546971516.dat","IMAGE_PATH":"/Users/upload/DataSets/DI210056/1_2_110_109_1599546971516.jpg","DATASET_CD":"DI210056","DATA_CD":"00000002","TAGS":[{"CLASS_DB_NM":"1536","COLOR":"#3D12CA","ACCURACY":0.9669204950332642},{"CLASS_DB_NM":"1535","COLOR":"#7499EA","ACCURACY":0.9603719711303711},{"CLASS_DB_NM":"1534","COLOR":"#964964","ACCURACY":0.8729372620582581}],"BASE_MDL":"DI20210061","TOTAL_FRAME":0,"FPS":0},{"FILE_PATH":"/Users/upload/DataSets/DI210056/1_2_111_110_1599546971516.dat","IMAGE_PATH":"/Users/upload/DataSets/DI210056/1_2_111_110_1599546971516.jpg","DATASET_CD":"DI210056","DATA_CD":"00000003","TAGS":[{"CLASS_DB_NM":"1534","COLOR":"#964964","ACCURACY":0.881903350353241},{"CLASS_DB_NM":"1536","COLOR":"#3D12CA","ACCURACY":0.8556084036827087},{"CLASS_DB_NM":"1535","COLOR":"#7499EA","ACCURACY":0.659853458404541}],"BASE_MDL":"DI20210061","TOTAL_FRAME":0,"FPS":0},{"FILE_PATH":"/Users/upload/DataSets/DI210056/1_2_112_111_1599546971516.dat","IMAGE_PATH":"/Users/upload/DataSets/DI210056/1_2_112_111_1599546971516.jpg","DATASET_CD":"DI210056","DATA_CD":"00000004","TAGS":[{"CLASS_DB_NM":"1535","COLOR":"#7499EA","ACCURACY":0.8774477243423462},{"CLASS_DB_NM":"1534","COLOR":"#964964","ACCURACY":0.851233184337616},{"CLASS_DB_NM":"1536","COLOR":"#3D12CA","ACCURACY":0.674436628818512}],"BASE_MDL":"DI20210061","TOTAL_FRAME":0,"FPS":0}]}`
    // req.body = JSON.parse(req.body)
    // //test
    // console.log("SEGSEGSEGF")
    // console.log(JSON.stringify(req.body))
    //1. 데이터셋 상태 변경
    //2. 프리트레인트 클래스 가져오기
    //3. 태그 저장
    // //4. 데이터 엘리먼트 수정
    // console.log("HERE SEG")
    // console.log("==============================================")
    // console.log("==================BIN======================")
    // console.log(JSON.stringify(req.body))
    // console.log("===========================================")
    const param = req.body.RESULT
    const DATASET_CD = param[0].DATASET_CD
    const DATA_CD = param[0].DATA_CD
    const BASE_MDL = param[0].BASE_MDL
    //데이터셋 상태 변경
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.param = {}
    option.param.DATASET_CD = DATASET_CD

    option.source = CC.MAPPER.BIN
    option.queryId = "initDataTags"
    await DH.executeQuery(option)

    option.param.DATA = []

    let tagDatas = []
    param.map((ele) => {
      tagDatas = tagDatas.concat(ele.TAGS)
      option.param.DATA.push({
        DATA_CD: ele.DATA_CD,
        ANNO_DATA: ele.FILE_PATH
      })
    })

    //데이터 엘리먼트 수정
    option.param.IS_ANNO = true
    option.queryId = "setUpdateDataElement"
    await DH.executeQuery(option)

    const classDBNames = tagDatas
      .map((ele) => ele.CLASS_DB_NM)
      .filter((v, i, s) => s.indexOf(v) === i)

    tagDatas = tagDatas
      .map((ele) => JSON.stringify(ele))
      .filter((v, i, s) => s.indexOf(v) === i)
    tagDatas = tagDatas.map((ele) => JSON.parse(ele))

    // pre train model list 가져와서 현재 BASE_MDL이 pre train model 인지 확인
    option.queryId = "getPrdTrainModelList"
    let preTrainList = await DH.executeQuery(option)
    const isBaseMdl = preTrainList.some((ele) => ele.BASE_MDL === BASE_MDL)

    option.param.DATA = classDBNames
    option.param.BASE_MDL = BASE_MDL

    let tagList = []
    if (option.param.DATA.length > 0 && isBaseMdl) {
      // 프리트레인드 클래스 가져오기
      option.queryId = "getPrePredictTag"
      tagList = await DH.executeQuery(option)
    } else if (option.param.DATA.length > 0 && !isBaseMdl) {
      // ai model class 가져오기
      option.queryId = "getAiModelTag"
      tagList = await DH.executeQuery(option)
    }
    option.param.DATA = []
    console.log("=====tagDatas======")
    console.log(tagDatas)
    console.log("=====tagList======")
    console.log(tagList)
    tagList.map((ele) => {
      let temp = tagDatas.find((item) => item.CLASS_DB_NM === ele.CLASS_DB_NAME)
      if (temp) {
        option.param.DATA.push({
          NAME: ele.CLASS_DP_NAME,
          CLASS_CD: ele.CLASS_CD,
          COLOR: temp.COLOR
        })
      }
    })
    if (option.param.DATA.length !== 0) {
      //데이터 테그 설정
      option.queryId = "setDataTags"
      await DH.executeQuery(option)
    }

    if (isBaseMdl) {
      //pre train model
      option.queryId = "getDataTags"
    } else {
      // ai model
      option.queryId = "getDataTagsByAiModel"
    }
    let savedTag = await DH.executeQuery(option)
    let tagInfo = {}
    savedTag.map((ele) => {
      tagInfo[ele.COLOR] = {}
      tagInfo[ele.COLOR].TAG_CD = ele.TAG_CD
      tagInfo[ele.COLOR].CLASS_DP_NAME = ele.CLASS_DP_NAME
    })

    logger.debug(
      `[${DATASET_CD}] AUTOLABEL PreProcessing Tag Info \n${JSON.stringify(
        tagInfo,
        null,
        1
      )}`
    )
    // //결과 파일 수정
    option.param.DATA = []
    const chFile = param.map((ele) => {
      return new Promise((resolve, reject) => {
        let uniqTagCds = []
        var rsFile = JSON.parse(String(fs.readFileSync(ele.FILE_PATH)))
        rsFile.POLYGON_DATA.map((frame) => {
          if (tagInfo[frame.COLOR] !== undefined) {
            try {
              frame.DATASET_CD = DATASET_CD
              frame.DATA_CD = ele.DATA_CD
              frame.TAG_CD = tagInfo[frame.COLOR].TAG_CD
              frame.TAG_NAME = tagInfo[frame.COLOR].CLASS_DP_NAME
              uniqTagCds.push(tagInfo[frame.COLOR].TAG_CD)
            } catch (error) {
              console.log("============error=========")
              console.log(error)
            }
          }
        })
        const unique = Array.from(new Set(uniqTagCds))
        option.param.DATA.push({
          DATA_CD: ele.DATA_CD,
          TAG_CNT: unique.length,
          ANNO_CNT: rsFile.POLYGON_DATA.length
        })
        // console.log(rsFile)
        fs.writeFileSync(ele.FILE_PATH, JSON.stringify(rsFile), {
          encoding: "utf8",
          flag: "w"
        })
        resolve(1)
      })
    })

    await Promise.all(chFile)

    option.source = CC.MAPPER.IMG_ANNO
    option.queryId = "setUpdateDataElementArray"
    await DH.executeQuery(option)

    // _sendEmail(
    //   "dmshin@weda.kr",
    //   "[BluAi] 예측 완료",
    //   "요청하신 데이터셋의 사전 예측이 완료되었습니다." +
    //     JSON.stringify(req.body)
    // )
    res.json({ status: 1 })

    option.source = CC.MAPPER.IMG_ANNO
    option.param = {}
    option.param.DATASET_CD = DATASET_CD
    option.param.DATASET_STS = "DONE"
    option.param.PID = -1
    option.queryId = "updateDataSetStatus"
    await DH.executeQuery(option)

    logger.info(`[${DATASET_CD}] AUTOLABEL PreProcessing Success`)
  })
)

module.exports = router
