import express, { response } from "express"
import si from "systeminformation"
import moment from "moment"
import rp from "request-promise"

import CC from "../lib/CommonConstants"
import DH from "../lib/DatabaseHandler"
import CL from "../lib/ConfigurationLoader"

const config = CL.getConfig()

const router = express.Router()

router.post("/:routeType", function(req, res, next) {
  const routeType = req.params.routeType
  switch (routeType) {
    case "setSystemResource":
      _setSystemResource(req, res)
      break
    case "getResourceLog":
      _getResourceLog(req, res)
      break
    case "getLocationResource":
      _getLocationResource(req, res)
      break
    case "getSystemResource":
      _getSystemResource(req, res)
      break
  }
})

async function _getSystemResource(req, res) {
  const currentLoad = new Promise((resolve, reject) => {
    si.currentLoad()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  const mem = new Promise((resolve, reject) => {
    si.mem()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  const fsSize = new Promise((resolve, reject) => {
    si.fsSize()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  const diskLayout = new Promise((resolve, reject) => {
    si.blockDevices()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  const graphics = new Promise((resolve, reject) => {
    si.graphics()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })

  Promise.all([currentLoad, mem, fsSize, diskLayout, graphics])
    .then(data => {
      const j = {
        cpuData: data[0],
        memoryData: data[1],
        diskData: data[2],
        diskLayout: data[3],
        graphicsData: data[4]
      }
      res.json(j)
    })
    .catch(err => {
      res.status(500).send({
        error: "DataBase Error",
        msg: String(err.message)
      })
    })
}

async function _getLocationResource(req, res) {
  let option = {}
  option.source = CC.MAPPER_NAME.SYSTEM
  option.nameSpace = "SystemMapper"
  option.queryId = "getLocation"
  option.param = req.body

  await DH.executeQuery(option)
    .then(data => {
      const serverAdr = data[0].SRV_ADDR
      const port = config.port
      let rpOption = {
        method: "POST",
        url: `http://${serverAdr}:${port}/api/system/getSystemResource`,
        json: true,
        encoding: null
      }
      // let rpOption = {
      //   method: "POST",
      //   url: `http://localhost:${port}/api/system/getSystemResource`,
      //   json: true,
      //   encoding: null
      // }
      rp(rpOption)
        .then(d => {
          res.json(d)
        })
        .catch(err => {
          res.status(500).send({
            error: "DataBase Error",
            msg: String(err.message)
          })
        })
    })
    .catch(err => {
      res.status(500).send({
        error: "DataBase Error",
        msg: String(err.message)
      })
    })
}

async function _setSystemResource(req, res) {
  let option = {}
  option.source = CC.MAPPER_NAME.SYSTEM
  option.nameSpace = "SystemMapper"
  option.queryId = "setSystemResource"
  option.param = req.body
  const currentLoad = new Promise((resolve, reject) => {
    si.currentLoad()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  const mem = new Promise((resolve, reject) => {
    si.mem()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  const fsSize = new Promise((resolve, reject) => {
    si.fsSize()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  const graphics = new Promise((resolve, reject) => {
    si.graphics()
      .then(data => {
        resolve(data)
      })
      .catch(error => reject(error))
  })
  Promise.all([currentLoad, mem, fsSize, graphics]).then(data => {
    const a = data[2].map(diskData => {
      return new Promise((resolve, reject) => {
        req.body.ALY_DTM = moment().format("YYYYMMDDHHmmss")
        req.body.SVR_LOC = CL.get("serverLocation")
        req.body.DISK_NM = diskData.fs
        req.body.CPU_UQTY = (100 - data[0].currentload_idle).toFixed(2)
        req.body.RAM_FULL_VOL = data[1].total
        req.body.RAM_UQTY = data[1].available
        req.body.RAM_USE_RTO = (
          (data[1].available / data[1].total) *
          100
        ).toFixed(2)
        req.body.DISK_FULL_VOL = diskData.size
        req.body.DISK_UQTY = diskData.used
        req.body.DISK_USE_RTO = diskData.use
        DH.executeQuery(option)
          .then(data => {
            resolve(data)
          })
          .catch(err => {
            reject(err)
          })
      })
    })
    Promise.all(a)
      .then(data => {
        res.json(data)
      })
      .catch(err => {
        res.status(500).send({
          error: "DataBase Error",
          msg: String(err.message)
        })
      })
  })
}

async function _getResourceLog(req, res) {
  let option = {}
  option.source = CC.MAPPER_NAME.SYSTEM
  option.nameSpace = "SystemMapper"
  option.queryId = "getResourceLog"
  option.param = req.body
  switch (req.body.TYPE) {
    case "CPU":
      option.param.COLUMN_TO_SELECT = "MAX(CPU_UQTY) CPU_UQTY"
      break
    case "Memory":
      option.param.COLUMN_TO_SELECT =
        "MAX(RAM_FULL_VOL) RAM_FULL_VOL ,MAX(RAM_UQTY) RAM_UQTY,MAX(RAM_USE_RTO) RAM_USE_RTO"
      break
    case "Disk":
      option.param.COLUMN_TO_SELECT = `LISTAGG(DISK_NM,'||') WITHIN GROUP(ORDER BY DISK_NM) AS DISK_NM,
        LISTAGG(DISK_FULL_VOL,'||') WITHIN GROUP(ORDER BY DISK_NM) AS DISK_FULL_VOL,
        LISTAGG(DISK_UQTY,'||') WITHIN GROUP(ORDER BY DISK_NM) AS DISK_UQTY,
        LISTAGG(DISK_USE_RTO,'||') WITHIN GROUP(ORDER BY DISK_NM) AS DISK_USE_RTO`
      break
    default:
      option.param.COLUMN_TO_SELECT = "*"
      break
  }
  DH.executeQuery(option)
    .then(data => {
      if (req.body.TYPE === "Disk") {
        option.queryId = "getDiskLogDistnct"
        DH.executeQuery(option)
          .then(d => {
            res.json({
              data: data,
              distinct: d
            })
          })
          .catch(e => {
            res.status(500).send({
              error: "DataBase Error",
              msg: String(e.message)
            })
          })
      } else {
        res.json(data)
      }
    })
    .catch(err => {
      res.status(500).send({
        error: "DataBase Error",
        msg: String(err.message)
      })
    })
}

module.exports = router
