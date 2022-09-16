import schedule from "node-schedule"
import CC from "./CommonConstants"
import CF from "./CommonFunction"
import CL from "./ConfigurationLoader"
import DH from "./DatabaseHandler"

var logger = require("../lib/Logger")(__filename)
import rp from "request-promise"
import fs from "fs"
import fe from "fs-extra"
import path from "path"
import moment from "moment"
import chalk from "chalk"
import si from "systeminformation"
import syncRequest from "sync-request"

class ScheduleerHandler {
  getScheduleList() {
    return JSON.stringify(schedule.scheduledJobs)
  }

  startCheckJob() {
    const cron = CL.get("checkSchedule")
    schedule.scheduleJob("CHECK_STATUS", cron, async () => {
      let option = {}
      option.source = CC.MAPPER.IS
      option.queryId = "getIsCamList"
      let list = await DH.executeQuery(option)
      list.map((ele) => {
        CF.multiHelthCheck(ele)
      })
      // logger.set("app", "s", "[SYS] Check InputSorce Status")
    })
  }

  startGpuCheckJob() {
    const config = CL.getConfig()
    const cron = CL.get("checkSchedule")
    schedule.scheduleJob("CHECK_STATUS", cron, async () => {
      try {
        let gpuInfo = await CF.sendRequestResLong(
          config.masterIp,
          config.masterPort,
          CC.URI.gpuInfo,
          {}
        )
        CF.resException(gpuInfo, CC.URI.gpuInfo)

        gpuInfo.map((ele, idx) => {
          ele.GPU_NAME = idx + "_" + ele.GPU_NAME
          ele.GPU_USED_RATE = ele.MEM_USED / ele.MEM_TOTAL
        })
        if (gpuInfo.length > 0) {
          let option = {}
          option.source = CC.MAPPER.SYS
          option.queryId = "setGpuResource"
          option.param = {}
          option.param.DATA = gpuInfo
          await DH.executeQuery(option)
        }
      } catch (error) {
        logger.error(`[GET GPU Resource] \n${error.stack}`)
      }
    })
  }
}

module.exports = new ScheduleerHandler()
