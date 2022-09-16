import express from "express"
import asyncHandler from "express-async-handler"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import gm from "gm"

import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"
var logger = require("../lib/Logger")(__filename)

const router = express.Router()
const config = CL.getConfig()
const CRN_USR = "testUser@test.co.kr"
const spawn = require("child_process").spawn

const _base64Encode = (file) => {
  var bitmap = fs.readFileSync(file)
  return Buffer.from(bitmap, "base64")
}

const _getImageInfo = (file) => {
  return new Promise((resolve, reject) => {
    gm(file).identify("%w", function (err, info) {
      console.log(info)
      var m,
        exif = {},
        re = /^(?:exif:)?(\w+)=(.+)$/gm
      while ((m = re.exec(info))) {
        exif[m[1]] = m[2]
      }
      console.log(exif)
      resolve(exif)
    })
  })
}

router.post(
  "/getVideoList",
  asyncHandler(async (req, res, next) => {
    let thumUrl = "/static/"

    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.queryId = "getImageList"
    option.param = req.body
    option.param.DATA_STATUS = "ORG"

    thumUrl += option.param.DATASET_CD

    let list = await DH.executeQuery(option)
    list.map((ele) => {
      let FILE_PATH
      const temp = ele.FILE_PATH.indexOf(option.param.DATASET_CD)
      if (temp > -1)
        FILE_PATH = ele.FILE_PATH.substr(
          option.param.DATASET_CD.length + temp,
          ele.FILE_PATH.length
        )

      ele.THUM =
        thumUrl +
        path.join(
          path.dirname(FILE_PATH),
          "THUM_" + path.parse(path.basename(FILE_PATH)).name + ".jpg"
        )
      ele.FILE_SIZE = ele.FILE_SIZE
      ele.IMG_WITH = "..."
      ele.IMG_HEIGHT = "..."
      ele.IMG_INFO = "..."
    })
    res.json(list)
  })
)

router.post(
  "/getVideo",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.VDO_ANNO
    option.queryId = "getVideo"
    option.param = req.body

    let vdoFile = await DH.executeQuery(option)
    vdoFile = vdoFile[0]
    let objectInfo = []
    if (vdoFile.IS_ANNO) {
      // let filePath = path.join(config.datasetPath, vdoFile.DATASET_CD)
      // filePath = path.join(filePath, "result")
      // filePath = path.join(filePath, vdoFile.DATA_CD + ".dat")
      objectInfo = JSON.parse(String(fs.readFileSync(vdoFile.ANNO_DATA)))
    }

    let url = "/static"
    url = path.join(url, vdoFile.DATASET_CD)
    url = path.join(url, vdoFile.FILE_RELPATH)

    vdoFile.FILE_URL = url
    vdoFile.ANNO_DATA = objectInfo
    res.json(vdoFile)
  })
)

router.post(
  "/getVideo_tmp",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.VDO_ANNO
    option.queryId = "getVideo"
    option.param = req.body

    let imgFile = await DH.executeQuery(option)
    imgFile = imgFile[0]
    const fileName = imgFile.FILE_NAME + imgFile.FILE_EXT
    const imgPath = path.dirname(imgFile.FILE_PATH)

    imgFile.IMG_MASK = null
    if (imgFile.DATA_STATUS === "MASKED")
      imgFile.IMG_MASK = _base64Encode(path.join(imgPath, "MASK_" + fileName))

    if (imgFile.ANNO_DATA !== null && imgFile.ANNO_DATA.length > 0) {
      imgFile.ANNO_DATA = imgFile.ANNO_DATA.substr(
        0,
        imgFile.ANNO_DATA.length - 1
      )
      const rects = imgFile.ANNO_DATA.split(";")
      imgFile.ANNO_DATA = []
      rects.map((ele) => {
        let temp = ele.split(",")
        imgFile.ANNO_DATA.push({
          DATA_CD: imgFile.DATA_CD,
          DATASET_CD: imgFile.DATASET_CD,
          POSITION: [
            { X: temp[0], Y: temp[1] },
            { X: temp[2], Y: temp[3] }
          ],
          TAG_CD: temp[4],
          COLOR: temp[5]
        })
      })
    } else imgFile.ANNO_DATA = []

    // imgFile.VIDEO_ORG = _base64Encode(imgFile.FILE_PATH)
    imgFile.VIDEO_TYPE =
      "data:video/" + imgFile.FILE_EXT.substr(1, imgFile.FILE_EXT.length) + ";"
    res.json(imgFile)
  })
)

router.post(
  "/getDataTags",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.queryId = "getDataTags"
    option.param = req.body
    let tagList = await DH.executeQuery(option)
    const objType = req.body.OBJECT_TYPE

    const promises = tagList.map((ele) => {
      return new Promise((resolve, reject) => {
        option.queryId = "getDataCategory"
        option.param.PARENTS_SEQ1 = ele.CATEGORY1
        option.param.PARENTS_SEQ2 = ele.CATEGORY2
        DH.executeQuery(option)
          .then((data) => {
            ele.CATEGORY1_LIST = []
            ele.CATEGORY2_LIST = []
            ele.CATEGORY3_LIST = []

            data.map((item) => {
              if (item.DEPTH === 0 && item.OBJECT_TYPE === objType) {
                ele.CATEGORY1_LIST.push({
                  label: item.CATEGORY_NAME,
                  value: item.CATEGORY_SEQ
                })
              }
              if (item.DEPTH === 1 && item.OBJECT_TYPE === objType) {
                ele.CATEGORY2_LIST.push({
                  label: item.CATEGORY_NAME,
                  value: item.CATEGORY_SEQ
                })
              }
              if (item.DEPTH === 2 && item.OBJECT_TYPE === objType) {
                ele.CATEGORY3_LIST.push({
                  label: `${item.CLASS_DP_NAME} (${item.CATEGORY_NAME}) - ${item.BASE_MDL}`,
                  value: item.CLASS_CD
                })
              }
            })
            resolve(1)
          })
          .catch((err) => {
            reject(err)
          })
      })
    })

    await Promise.all(promises).then((data) => {
      // console.log(data)
    })
    // console.log(tagList)
    res.json(tagList)
  })
)

router.post(
  "/setDataTag",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.queryId = "setDataTag"
    option.param = req.body
    await DH.executeQuery(option)
    res.json({ status: 1 })
  })
)

router.post(
  "/removeDataTag",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.queryId = "removeDataTag"
    option.param = req.body

    await DH.executeQuery(option)
    res.json({ status: 1 })
  })
)

router.post(
  "/getCategory",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.param = req.body

    if (
      option.param.CATEGORY1 !== undefined &&
      option.param.CATEGORY2 !== undefined
    )
      option.queryId = "getCategory3"
    else if (option.param.CATEGORY1 !== undefined)
      option.queryId = "getCategory2"
    else option.queryId = "getCategory1"

    let categoryList = await DH.executeQuery(option)
    res.json(categoryList)
  })
)

router.post(
  "/getImagePredict",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.param = req.body
    option.queryId = "getImage"

    let imgInfo = await DH.executeQuery(option)
    imgInfo = imgInfo[0]

    option.queryId = "getPreTraindClass"
    let classInfo = await DH.executeQuery(option)
    classInfo = classInfo[0]

    let item = []
    item.push(CC.BIN.miniPredictor)
    item.push(
      JSON.stringify({
        IMAGES: [
          {
            IMAGE_PATH: imgInfo.FILE_PATH,
            COLOR: null,
            RECT: null
          }
        ],
        MODE: 1,
        OBJECT_TYPE: option.param.OBJECT_TYPE,
        CLASS_DB_NM: classInfo.CLASS_DB_NAME,
        MDL_PATH: classInfo.MDL_PATH
      })
    )
    console.log(item)
    let result = await _runProcessSync(item)
    result = JSON.parse(result)
    res.json(result)
  })
)

router.post(
  "/setVideoAnnotation",
  asyncHandler(async (req, res, next) => {
    const param = req.body.ANNO_DATA
    let rectStr = ""

    let filePath = path.join(config.datasetPath, req.body.DATASET_CD)
    filePath = path.join(filePath, "result")
    filePath = path.join(filePath, req.body.DATA_CD + ".dat")

    fs.writeFileSync(filePath, JSON.stringify(param))
    let maxCnt = 0
    param.POLYGON_DATA.map((ele, idx) => {
      if (ele.length > 0) maxCnt = idx
    })
    let option = {}
    option.source = CC.MAPPER.IMG_ANNO
    option.param = {}
    option.param.DATASET_CD = req.body.DATASET_CD
    option.param.DATA_CD = req.body.DATA_CD
    option.param.ANNO_CNT = maxCnt
    option.param.ANNO_DATA = filePath
    option.param.IS_ANNO = true
    option.queryId = "setUpdateDataElement"

    await DH.executeQuery(option)

    res.json({ status: 1 })
  })
)

router.post(
  "/getTrackedObject",
  asyncHandler(async (req, res, next) => {
    let item = []
    item.push(CC.BIN.trackOBJECT)
    item.push(JSON.stringify(req.body))
    console.log("go")
    let result = await _runProcessSync(item)
    res.json(JSON.parse(result))
  })
)
function _runProcess(args) {
  const ls = spawn("python", args)
}
function _runProcessSync(args) {
  return new Promise((resolve, reject) => {
    const ls = spawn("python", args)
    let result = ""
    let Err = ""
    ls.stdout.on("data", function (data) {
      result += data
      // console.log(String(data))
    })

    ls.stderr.on("data", function (data) {
      Err += data
      // console.log(String(data))
    })
    ls.on("exit", function (code) {
      if (code === 0) {
        resolve(result)
      } else {
        reject(String(Err))
      }
    })
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
  "/getTrackResult",
  asyncHandler(async (req, res, next) => {
    //인 아웃은 민기랑 맞춤
    //시작값, 끝값 , 렉트, 클래스정보(태그정보- 컬러, DP네임 )
    // let file = req.body.FILE
    // let start_frame = req.body.START_FRAMELE
    // let reqData = req.body
    res.setTimeout(60 * 10 * 1000, function () {
      logger.error("Request has timed out.  in TACKER!!")
      res.send(408)
    })

    res.on("close", () => {
      logger.error("close Client !!!!")
    })

    let result
    try {
      let param = {}
      param = req.body
      // param.FILE_PATH = data
      // param.ARRAY = [{}, {}]
      // param.JSON_DATA = { test: "sdfsdf" }
      let res_Mother = await CF.sendRequestResLong(
        config.masterIp, //IP
        5638, //PORT
        "/tracker", //Dest
        param //Parm
      )
      result = res_Mother
    } catch (error) {
      result = { status: 0, msg: error }
    } finally {
      res.json(result)
    }
  })
)

router.post(
  "/getPredictResult",
  asyncHandler(async (req, res, next) => {
    const isReady = req.body.IS_READY
    let runPredict = false
    let returnCode = [{ status: 1 }]
    let result = {}
    let checkModel = false
    let modelFull = false
    let maxModelCnt = P_BUILD === "EE" ? config.maxModelCnt : 1
    try {
      result = await CF.sendRequestResLong(
        config.masterIp,
        config.masterPort,
        CC.URI.activePredictor,
        {}
      )
      if (result.length > maxModelCnt) modelFull = true
      result.map((ele) => {
        if (ele.AI_CD === req.body.AI_CD) checkModel = true
      })
    } catch (error) {
      returnCode = [{ status: 0, msg: error }]
    }

    if (isReady) {
      if (checkModel) runPredict = true
      else {
        if (modelFull) returnCode = [{ status: 2 }]
        else [{ status: 3 }]
      }
    } else {
      if (checkModel) [{ status: 1 }]
      else {
        if (modelFull) returnCode = [{ status: 2 }]
        else runPredict = true
      }
    }

    if (runPredict) {
      const magicRect = req.body.RECT
      let option = {}
      option.source = CC.MAPPER.IMG_ANNO
      option.param = req.body
      option.queryId = "getImage"
      let imgInfo = await DH.executeQuery(option)
      imgInfo = imgInfo[0]

      option.queryId = "getPreTraindClass"
      let classInfo = await DH.executeQuery(option)
      classInfo = classInfo[0]
      let param = {
        IMAGES: [
          {
            DATASET_CD: req.body.DATASET_CD,
            DATA_CD: req.body.DATA_CD,
            IMAGE_PATH: imgInfo.FILE_PATH,
            COLOR: option.param.COLOR,
            RECT: magicRect !== undefined ? magicRect : null,
            START_FRAME: req.body.START_FRAME,
            END_FRAME: req.body.END_FRAME
          }
        ],
        OBJECT_TYPE: option.param.OBJECT_TYPE,
        CLASS_DB_NM: classInfo.CLASS_DB_NAME,
        DATA_TYPE: "V",
        AI_TYPE: "PRE",
        AI_CD: classInfo.BASE_MDL,
        CALSS_CD: classInfo.CALSS_CD,
        MDL_PATH: classInfo.MDL_PATH,
        EPOCH: -1,
        IS_TEST: false
      }
      try {
        let mdlPid = await CF.sendRequestResLong(
          config.masterIp,
          config.masterPort,
          CC.URI.modelLoad,
          param
        )

        if (mdlPid.PID === 0) throw new Error("Model Load Fail")

        result = await CF.sendRequestResLong(
          config.masterIp,
          config.masterPort,
          CC.URI.miniPredictor,
          param
        )
      } catch (error) {
        result = { status: 0, msg: error }
      } finally {
        res.json(result)
      }
    } else res.json(returnCode)
  })
)

module.exports = router
