import CF from "./CommonFunction"
import CL from "../lib/ConfigurationLoader"
import C from "../lib/CommonConstants"
import x2j from "xml2json"
var logger = require("../lib/Logger")(__filename)

const config = CL.getConfig()

exports.setMasterConfig = async (configPath) => {
  let setResult = await CF.sendRequestResLong(
    config.masterIp,
    config.masterPort,
    C.URI.setMasterConfig,
    { CONFIG_PATH: configPath }
  )
  if (setResult.STATUS === 1) return true
  else return false
}

exports.setNvidiaSet = async () => {
  try {
    // let result = await CF.runProcess("nvidia-smi", ["-x", "-q"])
    // let xmlResult = result.stdout
    // xmlResult = JSON.parse(x2j.toJson(xmlResult))
    // const driverVersion = xmlResult.nvidia_smi_log.driver_version
    // const cudaVersion = xmlResult.nvidia_smi_log.cuda_version
    // console.log(`NVIDIA_VERSION: ${driverVersion}`)
    console.log(`NVIDI_DRIVER_LOAD`)
    //변환로직 추가
  } catch (error) {
    return false
  }
}

exports.initActiveProcess = async () => {
  try {
    await CF.exeQuery(C.Q.SYS, {}, "initDataSetSTS")
    logger.info(`[SYSTEM] init Dataset Process`)
    await CF.exeQuery(C.Q.SYS, {}, "initAiPrjSTS")
    logger.info(`[SYSTEM] init Ai Trainer Process`)
    await CF.exeQuery(C.Q.SYS, {}, "initServiceSTS")
    await CF.exeQuery(C.Q.SYS, {}, "initServiceCreateSTS")
    logger.info(`[SYSTEM] init Service Process`)
  } catch (error) {
    logger.error(`[SYSTEM] Init User ActiveProcess Fail \n${error}`)
    return false
  }
}
