import "@babel/polyfill"
import C from "./CommonConstants"
import CL from "./ConfigurationLoader"
import DH from "./DatabaseHandler"
import CF from "./CommonFunction"
// import ping from "net-ping"
import path from "path"
import request from "request"
import execa from "execa"

const config = CL.getConfig()

const logger = require("../lib/Logger")(__filename)

let BUILD = "CE"

const setBUILD = (build) => {
	BUILD = build
}

const getBUILD = () => {
	return BUILD
}

exports.exeQuery = async (source, param, queryId) => {
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

exports.getMdlPath = (aiCd, epoch) => {
	let mdl_path = path.join(config.aiPath, aiCd)
	mdl_path = path.join(mdl_path, String(epoch).padStart(4, 0))
	return mdl_path
}

exports.createProcess = async (cmd, param) => {
	return new Promise(async (resolve, reject) => {
		try {
			// execa.sync(cmd, param)
			execa(cmd, param)
			resolve(true)
		} catch (error) {
			logger.error(`Process Create Error \n${error.stderr}`)
			reject(error)
		}
	})
}

exports.runProcess = async (cmd, param) => {
	return new Promise(async (resolve, reject) => {
		try {
			let result = await execa(cmd, param)
			resolve(result)
		} catch (error) {
			logger.error(`Process Run Error`)
			reject(error)
		}
	})
}

exports.sendSoc = (data, type) => {
	process.send({ isSoc: 1, data, type })
}

exports.setSchedule = (data, type) => {
	process.send({ isSECH: 1, data, type })
}

exports.prints = async (IS_CD) => {
	logger.info(`Operation Service List Count[${SERVICE_AUTH_LIST}]`)
}

exports.popService = async (IS_CD) => {
	let temp = []
	SERVICE_AUTH_LIST.map((ele) => {
		if (ele.IS_CD !== String(IS_CD)) temp.push(ele)
	})
	console.log("==============")
	console.log(JSON.stringify(temp))
	process.send({ isService: "pop", data: IS_CD, listdata: temp })
}

exports.checkService = async (IS_CD, serviceAuth) => {
	logger.info(`Operation Service List Count[${SERVICE_AUTH_LIST.length}]`)

	let idx = SERVICE_AUTH_LIST.findIndex((ele) => ele.IS_CD === String(IS_CD))
	let param = {}
	param.IS_CD = IS_CD

	if (idx < 0) {
		//DB 조회
		let isInfo = await CF.exeQuery(C.Q.IS, param, "getActiveService")
		isInfo = isInfo[0]

		if (isInfo === undefined) return false
		isInfo.SERVICE_AUTH = String(isInfo.SERVICE_AUTH)

		isInfo.SERVICE_AUTH = serviceAuth
		if (isInfo.SERVICE_AUTH === null || isInfo.SERVICE_AUTH !== serviceAuth) {
			return false
		} else {
			let headers = await CF.exeQuery(C.Q.QP, param, "getHeaderList")
			let addService = {
				IS_CD: IS_CD,
				SERVICE_AUTH: serviceAuth,
				HEADERS: headers,
				IS_INFO: isInfo,
			}
			// headers = headers
			//   .filter((x) => x.IS_CLASS === 0)
			//   .map((el) => el.COLUMN_NAMES)
			// console.log(headers)

			// global.SERVICE_AUTH_LIST.push(addService)
			// process.send({ isService: "get", data: addService })
			process.send({ isService: "set", data: addService })
			return addService
		}
	} else {
		if (SERVICE_AUTH_LIST[idx].SERVICE_AUTH !== serviceAuth) return false
		else return SERVICE_AUTH_LIST[idx]
	}
}

exports.checkModel = async (AI_CD, isReady, EPOCH) => {
	let runPredict = false
	let returnCode = [{ status: 1 }]
	let checkModel = false
	let modelFull = false
	let maxModelCnt = P_BUILD === "EE" ? config.maxModelCnt : 1
	let result = await CF.sendRequestResLong(
		config.masterIp,
		config.masterPort,
		C.URI.activePredictor,
		{}
	)

	CF.resException(result, C.URI.activePredictor)

	if (result.length >= maxModelCnt) {
		logger.info(`Usable Model Full (${result.length}/${maxModelCnt})`)
		modelFull = true
	}
	result.map((ele) => {
		if (ele.AI_CD === AI_CD && ele.EPOCH === EPOCH) {
			checkModel = true
		}
	})

	if (isReady) {
		if (checkModel) runPredict = true
		else {
			if (modelFull) returnCode = [{ status: 2 }]
			else returnCode = [{ status: 3 }]
		}
	} else {
		if (checkModel) [{ status: 1 }]
		else {
			if (modelFull) returnCode = [{ status: 2 }]
			else runPredict = true
		}
	}
	return { code: returnCode, runPrc: runPredict }
}

exports.resException = (result, dst) => {
	if (result.STATUS !== undefined && result.STATUS === 0) {
		result.code = "Bin"
		result.dst = dst
		throw result
	} else return 1
}

exports.sendCurl = (msg, host) => {
	let options = {
		uri: host,
		method: "POST",
		body: msg,
		json: true, //json으로 보낼경우 true로 해주어야 header값이 json으로 설정됩니다.
	}
	// console.log("go aSync" + options.uri)
	request.post(options, (err, httpResponse, body) => {
		if (err) console.log(err)
		// else console.log(body.STATE)
	})
}

exports.sendRequest = (ip, port, rName, msg) => {
	let options = {
		uri: "http://" + ip + ":" + port + rName,
		method: "POST",
		body: [msg],
		json: true, //json으로 보낼경우 true로 해주어야 header값이 json으로 설정됩니다.
	}
	// console.log("go aSync" + options.uri)
	request.post(options, (err, httpResponse, body) => {
		if (err) console.log(err)
		// else console.log(body.STATE)
	})
}

exports.sendGetRequest = (ip, port, rName, msg) => {
	rName += "?"
	Object.keys(msg).map((key) => {
		rName += `${key}=${msg[key]}&`
	})

	rName = rName.substr(0, rName.length - 1)

	let url = "http://" + ip + ":" + port + rName
	console.log(url)

	let options = {
		uri: url,
		method: "GET",
		json: false, //json으로 보낼경우 true로 해주어야 header값이 json으로 설정됩s니다.
	}
	// console.log("go aSync" + options.uri)
	request.get(options, (err, httpResponse, body) => {
		if (err) {
			logger.error(`fail to Send Status ${url}\n${err.stack}`)
		}
		// else console.log(body.STATE)
	})
}

exports.sendRequestRes = (ip, port, rName, msg) => {
	return new Promise((resolve, reject) => {
		try {
			let options = {
				uri: "http://" + ip + ":" + port + rName,
				method: "POST",
				body: [msg],
				timeout: 5000,
				json: true, //json으로 보낼경우 true로 해주어야 header값이 json으로 설정됩니다.
			}
			logger.debug(`[REST_SHORT] ${options.uri}`)
			request.post(options, (err, httpResponse, body) => {
				if (err) {
					logger.error(err.stack)
					reject(err)
				} else resolve(body)
			})
		} catch (error) {
			logger.error(err.stack)
			reject(err)
		}
	})
}

exports.sendRequestResLong = (ip, port, rName, msg) => {
	return new Promise((resolve, reject) => {
		try {
			let options = {
				uri: "http://" + ip + ":" + port + rName,
				method: "POST",
				body: [msg],
				timeout: 0,
				json: true, //json으로 보낼경우 true로 해주어야 header값이 json으로 설정됩니다.
			}
			logger.debug(`[REST] ${options.uri}`)
			request.post(options, (err, httpResponse, body) => {
				if (err) {
					logger.error(err.stack)
					reject(err)
				} else resolve(body)
			})
		} catch (error) {
			logger.error(err.stack)
			reject(err)
		}
	})
}

const checkUrl = (session, url) => {
	return new Promise((resolve, reject) => {
		let target = url
		session.pingHost(target, function (error, target) {
			if (error) resolve(false)
			else resolve(true)
		})
	})
}

exports.multiHelthCheck = async (isInfo) => {
	try {
		console.log(isInfo)
	} catch (error) {}
}

// exports.multiHelthCheck2 = async (isInfo) => {
//   let srvIsAlive = true
//   let hwIsAlive = true
//   let errCode = ""
//   let msg = ""
//   let option = {}
//   if (isInfo.SRV_IP !== null) {
//     const session = ping.createSession({
//       sessionId: isInfo.IS_CD,
//       timeout: 1000
//     })
//     srvIsAlive = await checkUrl(session, isInfo.SRV_IP)
//   }
//   if (isInfo.HW_IP !== null) {
//     const session = ping.createSession({
//       sessionId: isInfo.IS_CD,
//       timeout: 1000
//     })
//     hwIsAlive = await checkUrl(session, isInfo.HW_IP)
//   }

//   let tmp = srvIsAlive && hwIsAlive
//   // console.log(
//   //   isInfo.HW_CD +
//   //     ":  SRV-" +
//   //     srvIsAlive +
//   //     "  HW-" +
//   //     hwIsAlive +
//   //     "         =" +
//   //     tmp
//   // )
//   if (tmp) {
//     option.source = CMAPPER.QP
//     option.queryId = "getISHWInfo"
//     option.param = isInfo
//     let is_info = await DH.executeQuery(option)
//     is_info = is_info[0]

//     if (is_info !== undefined && is_info.SRV_PID > 0) {
//       let srvPrarm = JSON.parse(JSON.stringify(is_info))
//       srvPrarm.PID = is_info.SRV_PID
//       try {
//         let resSrv = await CF.sendRequestRes(
//           is_info.SRV_IP,
//           5638,
//           "/check",
//           srvPrarm
//         )
//         if (resSrv.STATE !== 1) {
//           msg = "Somthing worng!, Plase restart your [ Server Module ]"
//           errCode = "120"
//         }
//       } catch (error) {
//         msg = "Somthing worng!, Plase restart your [ Server Module ]"
//         errCode = "120"
//       }
//     }
//     if (is_info !== undefined && is_info.HW_PID > 0) {
//       let hwParam = JSON.parse(JSON.stringify(is_info))
//       hwParam.PID = is_info.HW_PID
//       // console.log(hwParam)
//       // hwParam.SRV_IP = "192.168.0.5"
//       // hwParam.SRV_PORT = "10236"
//       console.log(hwParam.IS_CD + "         " + hwParam.SRV_IP)
//       try {
//         let resHw = await CF.sendRequestRes(
//           is_info.HW_IP,
//           5637,
//           "/check",
//           hwParam
//         )
//         // console.log("===================================================")
//         // console.log(hwParam)
//         // console.log("===================================================")
//         if (resHw.STATE !== 1) {
//           msg = "Somthing worng!, Plase reboot your [ HardWare ]"
//           errCode = "020"
//         }
//       } catch (error) {
//         msg = "Somthing worng!, Plase reboot your [ HardWare ]"
//         errCode = "020"
//       }
//     }
//   } else {
//     errCode = "010"
//     msg = "Somthing worng!, Plase check your [ HardWare(Connection Fail) ]"

//     option.source = CMAPPER.IS
//     option.queryId = "updateISStatus"
//     option.param = isInfo
//     option.param.STATUS = errCode === "200" ? true : false
//     option.param.STATUS_MSG = msg
//     option.param.ERR_MSG = ""
//     option.param.STATUS_CODE = errCode
//     await DH.executeQuery(option)
//   }

//   //업데이트 ㄲ
// }
