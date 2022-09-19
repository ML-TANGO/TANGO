import express from "express"
import asyncHandler from "express-async-handler"
import multer from "multer"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import gm from "gm"
import ncp from "ncp"
import ThumbnailGenerator from "video-thumbnail-generator"
import si from "systeminformation"

import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"

const router = express.Router()
const config = CL.getConfig()
const CRN_USR = "testUser@test.co.kr"
const spawn = require("child_process").spawn

router.post(
  "/getGpuInfo",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.SYS
    option.queryId = "getGpuResource"
    let list = await DH.executeQuery(option)
    let times = Array.from(new Set(list.map((ele) => ele.MRM_DTM)))
    let gpus = Array.from(new Set(list.map((ele) => ele.GPU_NM)))

    let resResult = {}
    resResult.GPU_LIST = gpus
    resResult.DATA = []

    times.map((ele) => {})

    times.map((ele) => {
      let timeEle = {}
      timeEle.DATE = ele
      list.map((item) => {
        if (item.MRM_DTM === ele) {
          timeEle[item.GPU_NM] = (item.GPU_USED_VOL / item.GPU_FULL_VOL) * 100
          timeEle[item.GPU_NM] = timeEle[item.GPU_NM].toFixed(3)
        }
      })
      resResult.DATA.push(timeEle)
    })

    res.json(resResult)
  })
)

router.post(
  "/getCurGpuInfo",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.SYS
    option.queryId = "getCurGpuResource"
    let list = await DH.executeQuery(option)
    let times = Array.from(new Set(list.map((ele) => ele.MRM_DTM)))
    let gpus = Array.from(new Set(list.map((ele) => ele.GPU_NM)))

    let resResult = {}
    resResult.GPU_LIST = gpus
    resResult.DATA = []

    times.map((ele) => {})

    times.map((ele) => {
      let timeEle = {}
      timeEle.DATE = ele
      list.map((item) => {
        if (item.MRM_DTM === ele) {
          timeEle[item.GPU_NM] = (item.GPU_USED_VOL / item.GPU_FULL_VOL) * 100
          timeEle[item.GPU_NM] = timeEle[item.GPU_NM].toFixed(3)
        }
      })
      resResult.DATA.push(timeEle)
    })

    res.json(resResult)
  })
)

router.post(
  "/getSystemInfo",
  asyncHandler(async (req, res, next) => {
    const currentLoad = new Promise((resolve, reject) => {
      si.currentLoad()
        .then((data) => {
          resolve(data)
        })
        .catch((error) => reject(error))
    })
    const mem = new Promise((resolve, reject) => {
      si.mem()
        .then((data) => {
          resolve(data)
        })
        .catch((error) => reject(error))
    })
    const fsSize = new Promise((resolve, reject) => {
      si.fsSize()
        .then((data) => {
          resolve(data)
        })
        .catch((error) => reject(error))
    })
    Promise.all([currentLoad, mem, fsSize]).then((data) => {
      const date = moment().format("YYYYMMDDHHmmss")
      const d = data[2].map((diskData) => {
        let j = {}
        j.ALY_DTM = date
        j.DISK_NM = diskData.fs
        j.CPU_UQTY = (100 - data[0].currentload_idle).toFixed(2)
        j.RAM_FULL_VOL = data[1].total
        j.RAM_UQTY = data[1].active
        j.RAM_USE_RTO = ((data[1].active / data[1].total) * 100).toFixed(2)
        j.DISK_FULL_VOL = diskData.size
        j.DISK_UQTY = diskData.used
        j.DISK_USE_RTO = diskData.use
        return j
      })
      let result = {}
      result.CPU_USED = d[0].CPU_UQTY
      result.RAM_USED = d[0].RAM_USE_RTO
      result.DISK_FULL = 0
      result.DISK_USED = 0
      d.map((ele) => {
        result.DISK_FULL += ele.DISK_FULL_VOL
        result.DISK_USED += ele.DISK_UQTY
      })
      result.DISK_USED = ((result.DISK_USED / result.DISK_FULL) * 100).toFixed(
        2
      )
      res.json(result)
    })
  })
)

router.post(
  "/getAiInfo",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.SYS
    option.queryId = "getAiPrjList"
    option.param = {}
    let AI_LIST = await DH.executeQuery(option)

    option.queryId = "getDataSetList"
    let datasetList = await DH.executeQuery(option)

    option.queryId = "getIsList"
    let isList = await DH.executeQuery(option)

    option.queryId = "getQiPrjList"
    let qi_prjList = await DH.executeQuery(option)

    res.json({
      AI_LIST: AI_LIST,
      DATASET_LIST: datasetList,
      IS_LIST: isList,
      PROJECT_LIST: qi_prjList
    })
  })
)

router.post(
  "/getSourceTreeMap",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.SYS
    option.queryId = "getSourceTreeMap"
    option.param = {}
    let list = await DH.executeQuery(option)

    res.json(list)
  })
)

router.post(
  "/getSidebarInfo",
  asyncHandler(async (req, res, next) => {
    let option = {}
    option.source = CC.MAPPER.SYS
    option.queryId = "getSidebarInfo"
    option.param = {}
    let list = await DH.executeQuery(option)

    res.json(list)
  })
)

module.exports = router
