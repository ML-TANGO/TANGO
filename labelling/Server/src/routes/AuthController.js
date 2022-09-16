import express from "express"
import asyncHandler from "express-async-handler"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import gm from "gm"
import ncp from "ncp"
import ThumbnailGenerator from "video-thumbnail-generator"
var crypto = require("crypto-js")
import si from "systeminformation"
import unzip from "node-unzip-2"
import tar from "tar"
import rimraf from "rimraf"

import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"
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

const _runProcessSync = (args) => {
  var cmd = require("node-cmd")
  return new Promise((resolve, reject) => {
    cmd.get(
      // `wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${fileCode}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=${fileCode}" -O ${downPath} && rm -rf cookies.txt`,
      args,
      function (error, success, stderr) {
        if (error) {
          resolve({ status: false, message: error })
        } else {
          resolve({ status: true })
        }
      }
    )
  })
}

const getFilesizeInBytes = (filename) => {
  var stats = fs.statSync(filename)
  var fileSizeInBytes = stats.size / 1024 / 1024
  return fileSizeInBytes
}

router.post("/getPretrainedModel", async (req, res, next) => {
  let mdlsPath = path.join(path.dirname(config.aiPath), "models")
  !fs.existsSync(mdlsPath) && fs.mkdirSync(mdlsPath)
  let fileCode = ""
  //let baseURL = "https://docs.google.com/uc?export=download&id="
  let modelKind = req.body.MDL_KIND
  let downPath = path.join(mdlsPath, modelKind + ".tar")

  const mainIP = "weda.kr"
  const mainPort = "80"
  const mainApi = "/api/getDownCode"

  let qrResult = await CF.sendRequestResLong(mainIP, mainPort, mainApi, {
    MDL_KIND: modelKind
  })
  qrResult = qrResult[0]
  console.log(qrResult)
  // if (modelKind === "detection") fileCode = "1ge2hJBfe_SRDqfYvNTodMe9wRrfmEgeo"
  // if (modelKind === "segmentation")
  //   fileCode = "1ErKf6aT9NZkhy56gantxfCBBxj1y39qo"
  // if (modelKind === "classification")
  //   fileCode = "15W8-bTmIMI9vDrWKm4TrxDa2QBnOIwFn"

  fileCode = qrResult.DOWN_CODE

  const wgetCmd = `wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${fileCode}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=${fileCode}" -O ${downPath} && rm -rf cookies.txt`

  // let result = await _runProcessSync(wgetCmd)

  let resResult = {}
  resResult.status = 0
  resResult.msg = "success"
  resResult.cmd = wgetCmd
  resResult.path = mdlsPath

  // downPath = "/Users/dmshin/Weda/BlueAi/test/detection.zip"

  if (!fs.existsSync(path.join(mdlsPath, modelKind))) {
    let result = await _runProcessSync(wgetCmd)
    // let result = {}
    // result.status = true

    if (result.status) {
      let fileSize = getFilesizeInBytes(downPath)
      if (fileSize > 10) {
        //압출 풀기
        resResult.status = 1
        const mdlDir = path.join(mdlsPath, modelKind)
        !fs.existsSync(mdlDir) && fs.mkdirSync(mdlDir)

        await new Promise((resolve, reject) => {
          fs.createReadStream(downPath)
            .pipe(
              tar.x({
                strip: 1,
                C: mdlDir
              })
            )
            .on("close", () => {
              resolve(true)
            })
            .on("error", (err) => {
              reject(false)
            })
        }).then((data) => {
          if (!data) {
            resResult.status = 0
            resResult.msg = "unzip Fail"
          }
        })
        rimraf.sync(downPath)
      } else resResult.msg = "file Size Error"
    } else resResult.msg = "file Down Error"
  } else {
    resResult.status = 1
    resResult.msg = "already exist " + modelKind
  }
  console.log("done")
  logger.info(
    `Init Model Download Result\n${JSON.stringify(resResult, null, 1)}`
  )
  res.json(resResult)
})

router.post("/getAuthCheck", async (req, res, next) => {
  const configPath = path.join(path.dirname(__dirname), "config")
  const licenseFile = path.join(configPath, ".license")

  try {
    if (fs.existsSync(licenseFile)) {
      let licenseInfo = {}
      let sysInfo = await si.system()
      let myLicense = String(fs.readFileSync(licenseFile))
      licenseInfo.uuid = sysInfo.uuid
      licenseInfo.serial = sysInfo.serial
      let license = crypto.MD5(JSON.stringify(licenseInfo)).toString()

      if (myLicense === license) {
        req.body.BUILD = req.body.BUILD === undefined ? "CE" : req.body.BUILD
        process.send({ isBuild: true, build: req.body.BUILD })
        res.json({ status: 1, msg: "Success" })
      } else throw new Error({ err: "license file dose not matched" })
    } else {
      res.json({ status: 1, msg: "Success" })
      // throw new Error({ err: "license file dose not exist" })
    }
  } catch (error) {
    // res.json({ status: 0, msg: error })
    res.json({ status: 1, msg: "Success" })
  }
})

router.post("/setAuthentication", async (req, res, next) => {
  const configPath = path.join(path.dirname(__dirname), "config")
  const licenseFile = path.join(configPath, ".license")

  const mainIP = "weda.kr"
  const mainPort = "80"
  const mainApi = "/api/authentication"

  let result = await CF.sendRequestResLong(mainIP, mainPort, mainApi, {
    LICENSE_CODE: req.body.LICENSE_CODE
  })

  if (result.status === 1) {
    let sysInfo = await si.system()
    let licenseInfo = {}
    licenseInfo.uuid = sysInfo.uuid
    licenseInfo.serial = sysInfo.serial
    let license = crypto.MD5(JSON.stringify(licenseInfo)).toString()

    if (license !== req.body.LICENSE_CODE) result.status = 0
    else
      fs.writeFileSync(licenseFile, req.body.LICENSE_CODE, {
        encoding: "utf8",
        flag: "w"
      })
  }
  res.json(result)
})

router.post("/setNewLicense", async (req, res, next) => {
  const configPath = path.join(path.dirname(__dirname), "config")
  const initDateFile = path.join(configPath, "initial_date.txt")
  let licenseInfo = {}
  let userInfo = {}

  let initDate = fs.statSync(initDateFile).birthtime
  let sysInfo

  sysInfo = await si.system()
  licenseInfo.uuid = sysInfo.uuid
  licenseInfo.serial = sysInfo.serial
  userInfo = req.body
  userInfo.PRODUCT_KIND = "BluAi"
  userInfo.INIT_DATE = initDate

  const mainIP = "weda.kr"
  const mainPort = "80"
  const mainApi = "/api/setnewlicense"

  let result = await CF.sendRequestResLong(mainIP, mainPort, mainApi, {
    licenseInfo: licenseInfo,
    userInfo: userInfo
  })
  res.json(result)
})

router.post("/login", login)
router.post("/refresh", refreshToken)
router.post("/logout", passport.authenticate("jwt", { session: false }), logout)
router.post(
  "/setuser",
  passport.authenticate("jwt", { session: false }),
  register
)
router.post(
  "/updateuser",
  passport.authenticate("jwt", { session: false }),
  changepw
)

router.post(
  "/checkuser",
  passport.authenticate("jwt", { session: false }),
  async (req, res, next) => {
    let option = {
      source: CC.MAPPER.AUTH,
      queryId: "checkUser",
      param: req.body
    }
    let result = await DH.executeQuery(option)
    res.json(result)
  }
)

router.post(
  "/getusers",
  passport.authenticate("jwt", { session: false }),
  async (req, res, next) => {
    let option = {
      source: CC.MAPPER.AUTH,
      queryId: "getUsers",
      param: req.body
    }
    let result = await DH.executeQuery(option)
    res.json(result)
  }
)

router.post(
  "/deleteuser",
  passport.authenticate("jwt", { session: false }),
  async (req, res, next) => {
    let option = {
      source: CC.MAPPER.AUTH,
      queryId: "deleteUser",
      param: req.body
    }
    await DH.executeQuery(option)
    res.json({ status: 1 })
  }
)

module.exports = router
