import express from "express"
import asyncHandler from "express-async-handler"
import multer from "multer"
import moment, { now } from "moment"
import path from "path"
import fs from "fs"
import fse from "fs-extra"
import rimraf from "rimraf"
import sizeOf from "image-size"

import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"
var logger = require("../lib/Logger")(__filename)

const router = express.Router()
const config = CL.getConfig()
const CRN_USR = "testUser@test.co.kr"
let totalNum = 0

let storage = multer.diskStorage({
	destination: (req, file, cb) => {
		let fileNames = file.originalname.replace(/_@_/g, "/")
		const dirPath = path.join(config.tempPath, path.dirname(fileNames))
		!fs.existsSync(dirPath) &&
			fs.mkdirSync(dirPath, {
				recursive: true,
			})

		cb(null, path.join(config.tempPath, path.dirname(fileNames)))
	},
	filename: (req, files, cb) => {
		let savedFile = files.originalname.replace(/ /g, "")
		savedFile = path.basename(savedFile.replace(/_@_/g, "/"))
		cb(null, savedFile) // cb 콜백함수를 통해 전송된 파일 이름 설정
	},
})

const upload = multer({ storage: storage })

router.post(
	"/upload",
	upload.array("file"),
	asyncHandler(async (req, res, next) => {
		const fileList = JSON.parse(req.body.dir)
		let result = []
		totalNum += fileList.length
		logger.info(`[UPLOAD] Uploaded: ${totalNum}, current: ${fileList.length}`)
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
	let option = {}
	let rollback = {}
	rollback.orgData = {}
	rollback.newData = {}
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

		option.source = CC.MAPPER.DATASET
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
	if (req.body.OBJECT_TYPE === "C") {
		const promises = req.body.files.map((ele) => {
			return new Promise((resolve, reject) => {
				if (ele.isNew) {
					try {
						let classDir = path.join(tempDir, ele.base)
						!fs.existsSync(classDir) && fs.mkdirSync(classDir)

						let imgPath = path.join(classDir, ele.path)

						fs.renameSync(path.join(tempDir, ele.path), imgPath)
						ele.path = path.join(ele.base, ele.path)
					} catch (error) {
						logger.error(error)
					}
				} else {
					if (!ele.path.includes(ele.base)) {
						try {
							let classDir = path.join(datasetDir, ele.base)
							!fs.existsSync(classDir) && fs.mkdirSync(classDir)

							let imgPath = path.join(classDir, ele.name)

							fs.renameSync(ele.path, imgPath)
							ele.prevPath = ele.path
							ele.path = path.join(classDir, ele.name)

							changed.push(ele)
						} catch (error) {
							logger.error(error)
						}
					}
				}
				resolve("")
			})
		})

		await Promise.all(promises)
	}

	let fileList = req.body.fileList
	fileList.map((ele) => {
		ele.path = ele.path.replace(/ /g, "")
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

	res.json({ status: 1 })
	req.body.changed = changed
	_createDataSets(req.body, dataCd)
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
		logger.info(`[REMOVE] Removing Dataset.... [${data.DATASET_CD}]`)
		const datasetDir = path.join(config.datasetPath, data.DATASET_CD)
		fs.existsSync(datasetDir) && rimraf.sync(datasetDir)
		const deploymentDir = path.join(
			config.deploymentPath,
			data.TITLE.replace(/[^\w\d\uAC00-\uD7AF]/g, "_")
		)
		fs.existsSync(deploymentDir) && rimraf.sync(deploymentDir)
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
		logger.info(`[REMOVE] Remove DataSet Success [${data.DATASET_CD}]`)
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
	try {
		let resResult = await CF.sendGetRequest("0.0.0.0", "10236", "/test", {
			project_id: "testtest",
			user_id: "dkdkdkfekjfj",
		})
	} catch (error) {
		logger.error(error)
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
					SAVE_PATH: thum,
				},
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
			TAG_CD: 0,
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
	res.json({ status: 1 })
	req.body.DATASET_DIR = datasetDir
	_createDataSets(req.body, 0)
})

router.post("/setNewDataSets", async (req, res, next) => {
	let option = {}
	let rollback = {}
	rollback.orgData = {}
	rollback.newData = {}

	// 데이터셋 코드 생성
	logger.info("[CREATE] Register new dataset code")
	// 발급
	option.source = CC.MAPPER.DATASET
	option.queryId = "setNewDataSetNumber"
	option.param = req.body
	option.param.YEAR = moment().format("YY")
	await DH.executeQuery(option)
	// 조회
	option.queryId = "getNewDataSetNumber"
	let cd = await DH.executeQuery(option)
	req.body.DATASET_CD = cd[0].DATASET_NUMBER
	logger.info(`[CREATE] DATASET_CD: ${req.body.DATASET_CD}`)

	// 파일 이동
	// ----------------------------------------------------------------------
	logger.info("[CREATE] Make file structure")
	const datasetDir = path.join(config.datasetPath, req.body.DATASET_CD)
	const tempDir = path.join(config.tempPath, req.body.uuid)
	req.body.DATASET_DIR = datasetDir
	if (req.body.OBJECT_TYPE === "C") {
		// create folder
		req.body.tags.forEach((tag) => {
			let datasetSubPath = path.join(datasetDir, tag.label)
			if (!fs.existsSync(datasetSubPath)) {
				fs.mkdirSync(datasetSubPath, { recursive: true })
			}
		})
		req.body.files.forEach((file) => {
			fs.renameSync(
				path.join(tempDir, file.path),
				path.join(datasetDir, file.base, file.name)
			)
		})
	} else {
		fs.renameSync(tempDir, datasetDir)
	}

	logger.info(
		`[CREATE] DATASET_ROOT_PATH: ${req.body.DATASET_DIR}, TEMP_ROOT_PATH: ${tempDir}`
	)

	// if(data.OBJECT_TYPE === "C" &&  req.body.files.find(ff=>ff.)){

	// }

	//데이터셋 메인 썸네일 생성
	// ----------------------------------------------------------------------
	logger.info("[CREATE] Make main thumbnail")
	const imageExtensions = [".jpg", ".jpeg", ".png", ".gif"]
	let selected = req.body.files.find((ff) =>
		imageExtensions.includes(path.extname(ff.name).toLowerCase())
	)
	let img = null
	// classification
	if (req.body.OBJECT_TYPE === "C") {
		img = path.join(datasetDir, selected.base, selected.name)
	} else {
		if (selected.base === "untagged") {
			img = path.join(datasetDir, selected.name)
		} else {
			img = path.join(datasetDir, selected.base, selected.name)
		}
	}

	let thum = path.join(
		datasetDir,
		"THUM_" + path.parse(path.basename(selected.name)).name + ".jpg"
	)
	logger.info(`[CREATE] Thumbnail image path: ${thum}`)
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
					SAVE_PATH: thum,
				},
			]
		)
		if (resResult.STATUS === 0) throw resResult.ERROR_FILE

		logger.info(`[CREATE] Query main thumbnail`)
		option.param.DATA = []
		option.param.DATA.push({
			DATASET_CD: req.body.DATASET_CD,
			DATA_CD: "T0000000",
			DATA_STATUS: "THUM",
			FILE_NAME: path.parse(thum).name,
			FILE_EXT: path.parse(thum).ext,
			FILE_TYPE: req.body.DATA_TYPE,
			FILE_PATH: thum,
			FILE_RELPATH: path.basename(thum),
			FILE_SIZE: 1,
			FPS: 0,
			TAG_CD: 0,
		})

		option.queryId = "setDataElement"
		await DH.executeQuery(option)
	} catch (error) {
		logger.error(error)
		res.json({ status: 0, err: error, msg: "썸네일 생성 실패" })
		rollbackDataSet(rollback.orgData, rollback.newData)
		return
	}

	// 데이터셋 생성
	logger.info("[CREATE] Create dataset")
	logger.info(`[CREATE] Query new dataset`)
	option.param.DATASET_CD = req.body.DATASET_CD
	option.param.CRN_USR = req.body.USER_ID || "user"
	option.param.THUM_NAIL_CD = "T0000000"
	option.param.CATEGORY1 = "USER"
	option.queryId = "setNewDataSet"
	await DH.executeQuery(option)

	logger.info(`[CREATE] Query dataset status`)
	option.source = CC.MAPPER.IMG_ANNO
	option.param = {}
	option.param.DATASET_CD = req.body.DATASET_CD
	option.param.DATASET_STS = "CREATE"
	option.queryId = "updateDataSetStatus"
	await DH.executeQuery(option)

	logger.info(`[CREATE] Create dataset`)
	_createDataSets(req.body, 0)

	res.json({ status: 1 })

	logger.info("[CREATE] Check Project manager")
	option.source = CC.MAPPER.TANGO
	option.queryId = "getProjectInfo"
	let list = await DH.executeQuery(option)

	if (list[0] !== undefined) {
		logger.info("send status request to Project Manager")
		try {
			let resResult = await CF.sendGetRequest(
				"project_manager",
				"8085",
				"/status_request",
				{
					project_id: list[0].PROJECT_ID,
					user_id: list[0].USER_ID,
				}
			)
		} catch (error) {
			logger.error(error)
		}
	} else
		logger.info("Fail to send status request to Project Manager : not started")
})

const _makeVideoThumbnails = async (data) => {
	logger.info("Video Data Processing...")
	let tmpFile = []
	data.files.map((ele) => {
		tmpFile.push({
			PATH: path.join(data.DATASET_DIR, ele.path),
			FILE_NAME: ele.name,
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
		data.files.map((file) => {
			if (ele.FILE_NAME === file.name) file.FPS = ele.FPS
		})
	})
	//비디오 일 경우 썸네일 전체 생성
	const promises = data.files.map((ele) => {
		return new Promise((resolve, reject) => {
			const elePath = path.join(data.DATASET_DIR, ele.path)
			let video = "THUM_" + path.parse(ele.path).name + ".jpg"
			video = path.join(data.DATASET_DIR, video)
			const result = CF.sendRequestResLong(
				config.masterIp,
				config.masterPort,
				CC.URI.makeThumnail,
				[
					{
						MDL_TYPE: data.DATA_TYPE,
						PATH: elePath,
						SAVE_PATH: video,
					},
				]
			).catch((err) => {
				reject(err)
			})
			resolve(result)
		})
	})

	await Promise.all(promises)
}

const _makeImageThumbnails = async (data) => {
	let thumFiles = []
	const fileCount = data.files.length
	data.files.map((ele, eleIdx) => {
		let filePath = data.DATASET_DIR
		filePath = path.join(filePath, ele.path)
		let savePath = path.dirname(filePath)
		savePath = path.join(savePath, "THUM_" + ele.name)
		savePath = savePath.replace(/ /g, "")
		filePath = filePath.replace(/ /g, "")
		thumFiles.push({
			MDL_TYPE: data.DATA_TYPE,
			PATH: filePath,
			SAVE_PATH: savePath,
		})
		logger.info(`[DATASET] (${eleIdx}/${fileCount}) ${savePath}`)
	})

	await CF.sendRequestResLong(
		config.masterIp,
		config.masterPort,
		CC.URI.makeThumnail,
		thumFiles
	).catch((err) => {
		throw new Error(`Thumnail Network Fail ${err.stack}`)
	})
}

const _setTagInfo = async (data) => {
	let option = {}
	option.source = CC.MAPPER.DATASET

	option.queryId = "getExistTag"
	option.param = { DATASET_CD: data.DATASET_CD }
	let existTags = await DH.executeQuery(option)

	existTags = existTags.map((em) => em.NAME)

	let newTags = [
		...new Set(
			data.files
				.filter((ff) => existTags.indexOf(ff.base) === -1)
				.map((ffm) => ffm.base)
		),
	]
	if (newTags.length > 0) {
		option.queryId = "setNewTag"
		option.param = { TAG: [] }
		newTags.map((nm) => {
			option.param.TAG.push({
				DATASET_CD: data.DATASET_CD,
				NAME: nm,
				CLASS_SUFFIX: nm,
			})
		})
		await DH.executeQuery(option)
	}
	option.queryId = "getExistTag"
	option.param.DATASET_CD = data.DATASET_CD
	return await DH.executeQuery(option)
}

const _setFileInfo = async (data, dataCd, tagInfo) => {
	let chunkSize = 100
	let option = {}
	option.source = CC.MAPPER.DATASET
	option.queryId = "setDataElement"
	option.param = { ...data, DATA: null }
	let fileInfo = []
	let newFiles = data.fileList ? data.fileList : data.files
	newFiles.map(async (ele, idx) => {
		let tempEle = {
			DATASET_CD: data.DATASET_CD,
			DATA_CD: String(dataCd + idx).padStart(8, 0),
			DATA_STATUS: "ORG",
			FILE_NAME: path.parse(ele.path).name,
			FILE_EXT: path.parse(ele.path).ext,
			FILE_TYPE: data.DATA_TYPE,
			FILE_PATH:
				data.OBJECT_TYPE !== "C" && ele.base === "untagged"
					? path.join(data.DATASET_DIR, ele.name)
					: path.join(data.DATASET_DIR, ele.base, ele.name),
			FILE_RELPATH:
				data.OBJECT_TYPE !== "C" && ele.base === "untagged"
					? ele.name
					: path.join(ele.base, ele.name),
			FILE_SIZE: ele.size,
			FPS: ele.FPS === undefined ? 0 : ele.FPS,
			TAG_CD: 0,
		}
		if (tagInfo) {
			const tag = tagInfo.find((item) => item.NAME === ele.base)
			tempEle.TAG_CD = tag.TAG_CD
		}

		fileInfo.push(tempEle)
	})

	for (let i = 0; i < fileInfo.length; i += chunkSize) {
		const chunk = fileInfo.slice(i, i + chunkSize)
		logger.info(
			`[DATASET] Insert Data Element ${i + chunk.length}/${fileInfo.length}`
		)
		option.param.DATA = chunk
		await DH.executeQuery(option)
	}
	data.DATA = fileInfo
	if (data.changed !== undefined && data.changed.length > 0) {
		let DATAS = []

		data.changed.map((ele) => {
			let tempData = {}
			tempData.FILE_NAME = path.parse(ele.name).name
			tempData.FILE_PATH = ele.path
			tempData.FILE_PREV_PATH = ele.prevPath
			let changedTag = tagInfo
				? tagInfo.find((tag) => tag.NAME === ele.base)
				: null
			tempData.TAG_CD = changedTag ? changedTag.TAG_CD : 0
			DATAS.push(tempData)
		})

		let optionTemp = {}
		optionTemp.source = CC.MAPPER.DATASET
		optionTemp.param = {
			DATASET_CD: data.DATASET_CD,
			DATAS,
		}
		optionTemp.queryId = "setBulkUpdateDataElement"
		logger.info(`-- Query update changes`)
		await DH.executeQuery(optionTemp)
	}
}

const _createDataSets = async (data, dataCd) => {
	let option = {}
	option.source = CC.MAPPER.DATASET
	option.param = data

	let tagInfo = null

	try {
		// 썸네일 생성
		if (data.OBJECT_TYPE !== "C") {
			logger.info(`[DATASET] Create thumbnail`)
			if (data.DATA_TYPE === "V") {
				await _makeVideoThumbnails(data)
			} else if (data.DATA_TYPE === "I") {
				await _makeImageThumbnails(data)
			}
		}
		// Classification인 경우 Tag등록
		else {
			logger.info(`[DATASET] Create tag (CLS)`)
			tagInfo = await _setTagInfo(data)
		}
		// 파일 등록
		logger.info(`[DATASET] Query file list`)
		await _setFileInfo(data, dataCd, tagInfo)

		//데이터 임폴트인 경우
		let splitOrgData = option.param.DATA.map((el) => el)
		if (data.IMPORT_TYPE === "COCO") {
			let argData = {
				DATASET_CD: data.DATASET_CD,
				PURPOSE_TYPE: data.OBJECT_TYPE,
				FILE_INFO: [],
				BASE_PATH: data.DATASET_DIR,
			}

			option.param.DATA = splitOrgData

			data.DATA.map((ele) => {
				argData.FILE_INFO.push({
					FILE_PATH: ele.FILE_PATH,
					DATA_CD: ele.DATA_CD,
				})
			})

			let argPath = path.join(data.DATASET_DIR, "arg.dat")

			fs.writeFileSync(argPath, JSON.stringify(argData), {
				encoding: "utf8",
				flag: "w",
			})
			logger.info("[DATASET] Import Data Processing...")
			let importResult = await CF.runProcess("python", [
				CC.BIN.importData,
				argPath,
			])

			if (importResult.stderr === "") {
				importResult = JSON.parse(importResult.stdout)

				const classList = importResult.CLASS_INFO

				let importOption = {}
				importOption.param = {}
				importOption.source = CC.MAPPER.BIN
				importOption.param.DATASET_CD = data.DATASET_CD
				importOption.param.DATA = []
				classList.map((ele) => {
					importOption.param.DATA.push({
						DATASET_CD: data.DATASET_CD,
						NAME: ele.TAG_NAME,
						COLOR: ele.COLOR,
					})
				})
				importOption.queryId = "setDataTags"
				if (importOption.param.DATA.length > 0) {
					await DH.executeQuery(importOption)
					importOption.source = CC.MAPPER.BIN
					importOption.param.DATA = []
					data.DATA.map((ele) => {
						let fn = ele.FILE_PATH
						let ext = ele.FILE_EXT

						let an = fn.substr(0, fn.length - ext.length)
						an = an + ".dat"

						if (ext !== ".json") {
							importOption.param.DATA.push({
								DATA_CD: ele.DATA_CD,
								ANNO_DATA: an,
								ANNO_CNT: 1,
								TAG_CNT: 1,
							})
						}
					})

					//모든 태그 변경
					importOption.source = CC.MAPPER.DATASET
					importOption.queryId = "getExistTag"
					let existTagList = await DH.executeQuery(importOption)

					existTagList.map((ele) => {
						let findTag = classList.find((item) => item.TAG_NAME === ele.NAME)
						ele.orgTagCd = findTag.TAG_CD
					})

					logger.info("[DATASET] Imported Class Data Mapping...")
					const totalNum = importOption.param.DATA.length
					const chFile = importOption.param.DATA.map((ele, eleIdx) => {
						return new Promise((resolve, reject) => {
							let filePath = ele.ANNO_DATA
							logger.info(
								`[DATASET] [${
									eleIdx + 1
								}/${totalNum}]Class mapping completed. Path=${filePath}`
							)
							if (fs.existsSync(filePath)) {
								var rsFile = JSON.parse(String(fs.readFileSync(filePath)))
								rsFile.POLYGON_DATA.map((frame) => {
									if (frame.TAG_CD != undefined && frame.TAG_CD != null) {
										let findTag = existTagList.find(
											(item) => item.orgTagCd === frame.TAG_CD
										)
										frame.TAG_CD = findTag.TAG_CD
									}
								})

								ele["ANNO_CNT"] = rsFile.POLYGON_DATA.length
								ele["TAG_CNT"] = [
									...new Set(rsFile.POLYGON_DATA.map((pm) => pm.TAG_CD)),
								].length
								fs.writeFileSync(filePath, JSON.stringify(rsFile), {
									encoding: "utf8",
									flag: "w",
								})
							}
							resolve(1)
						})
					})

					await Promise.all(chFile)

					//데이터 엘리먼트 수정
					importOption.source = CC.MAPPER.BIN
					importOption.param.IS_ANNO = true
					importOption.queryId = "setUpdateDataElementCnt"
					await DH.executeQuery(importOption)

					importOption.queryId = "removeJson"
					await DH.executeQuery(importOption)
					logger.info("[DATASET] Imported Cass Data Mapping Done!")
				}
			}
		}

		if (data.AUTO_TYPE === "Y") {
			data.AUTO_EPOCH = data.EPOCH === undefined ? -1 : data.EPOCH
			option.source = CC.MAPPER.DATASET
			option.param = {}
			option.param.DATASET_CD = data.DATASET_CD
			option.param.AUTO_ACC = data.AUTO_ACC
			option.param.AUTO_MODEL = data.AUTO_MODEL
			option.param.AUTO_EPOCH = data.AUTO_EPOCH
			option.param.AUTO_TYPE = data.AUTO_TYPE
			option.queryId = "setUpdateDataset"
			await DH.executeQuery(option)

			await _prePredictAll(data.DATASET_CD, data, data.AUTO_ACC)
		} else {
			option.source = CC.MAPPER.IMG_ANNO
			option.param = {}
			option.param.DATASET_CD = data.DATASET_CD
			option.param.DATASET_STS = "DONE"
			option.queryId = "updateDataSetStatus"
			await DH.executeQuery(option)
		}
	} catch (error) {
		logger.error(`[${data.DATASET_CD}] Create Fail \n${error.stack}`)
		option.source = CC.MAPPER.DATASET
		option.param = {}
		option.param.DATASET_CD = data.DATASET_CD
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
				RECT: null,
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
		AUTO_ACC: autoAcc,
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
			nodes: [],
		})
		dataSetTree.push({
			key: "USER",
			label: "USER DATASET",
			nodes: [],
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
				selectable: false,
			})
		})

		res.json(dataSetTree)
	})
)

router.post(
	"/checkDirExist",
	asyncHandler(async (req, res, next) => {
		let deploymentPath = path.join(config.deploymentPath, req.body.datasetName)
		res.json({ result: fs.existsSync(deploymentPath) })
	})
)
router.post(
	"/deployDataSet",
	asyncHandler(async (req, res, next) => {
		const datasetName = req.body.datasetName
		const datasetCd = req.body.datasetCd
		const split = req.body.split
		const objectType = req.body.type

		let option = { param: {} }
		option.source = CC.MAPPER.DATASET
		option.queryId = "updateDataSetStatus"
		option.param.DATASET_CD = datasetCd
		option.param.DATASET_STS = "CREATE"
		await DH.executeQuery(option)

		const datasetTypeName = (index, total) => {
			const targetPercentage = (index / total) * 100
			if (targetPercentage > parseInt(split[0], 10) + parseInt(split[1], 10))
				return "test"
			else if (targetPercentage > parseInt(split[0], 10)) return "valid"
			else return "train"
		}
		const normalizeCoordinates = (x1, y1, x2, y2, imageWidth, imageHeight) => {
			return {
				x: (x1 + x2) / 2 / imageWidth,
				y: (y1 + y2) / 2 / imageHeight,
				width: Math.abs(x2 - x1) / imageWidth,
				height: Math.abs(y2 - y1) / imageHeight,
			}
		}

		try {
			// create working directory
			const deploymentPath = path.join(config.deploymentPath, datasetName)
			const originalPath = path.join(config.datasetPath, datasetCd)
			!fs.existsSync(deploymentPath) &&
				fs.mkdirSync(deploymentPath, { recursive: true })
			// classification
			if (objectType === "C" || objectType === "S") {
				fse.copySync(originalPath, deploymentPath)
			}
			// detection & segmentation YOLOV7 type
			else {
				// get image & anno file list from database
				option.source = CC.MAPPER.DATASET
				option.queryId = "getFileList"
				option.param.DATASET_CD = datasetCd
				let fileList = await DH.executeQuery(option)

				// add DATASET_TYPE value  to split data (train/val/test)
				const fileCount = fileList.length
				logger.info(`File count: ${fileCount}`)
				fileList = fileList.map((fm, fmIdx) => ({
					...fm,
					DATASET_TYPE: datasetTypeName(fmIdx, fileCount),
				}))

				// get tag list
				option.queryId = "getExistTag"
				option.param.DATASET_CD = datasetCd
				const tagList = await DH.executeQuery(option)
				const tagNames = tagList.map((tm) => tm.NAME)

				// create data.yaml (category, category count, train_path, val_path)
				const dataYamlPath = path.join(deploymentPath, "data.yaml")
				let dataYaml = `# train and valid data\n`
				dataYaml += `train: ./train/images\n`
				dataYaml += `valid: ./valid/images\n\n`
				dataYaml += `# number of classes\n`
				dataYaml += `nc: ${tagList.length}\n\n`
				dataYaml += `# class names\n`
				dataYaml += `names: ${JSON.stringify(
					tagList.map((tm) => tm.NAME)
				).replace(/\"/g, "'")}`
				fs.writeFileSync(dataYamlPath, dataYaml, {
					encoding: "utf8",
					flag: "w",
				})
				// make dir (train_path/images,train_path/labels)
				let datasetTypePath = path.join(deploymentPath, "train", "images")
				!fs.existsSync(datasetTypePath) &&
					fs.mkdirSync(datasetTypePath, { recursive: true })
				datasetTypePath = path.join(deploymentPath, "train", "labels")
				!fs.existsSync(datasetTypePath) &&
					fs.mkdirSync(datasetTypePath, { recursive: true })
				datasetTypePath = path.join(deploymentPath, "valid", "images")
				!fs.existsSync(datasetTypePath) &&
					fs.mkdirSync(datasetTypePath, { recursive: true })
				datasetTypePath = path.join(deploymentPath, "valid", "labels")
				!fs.existsSync(datasetTypePath) &&
					fs.mkdirSync(datasetTypePath, { recursive: true })
				datasetTypePath = path.join(deploymentPath, "test", "images")
				!fs.existsSync(datasetTypePath) &&
					fs.mkdirSync(datasetTypePath, { recursive: true })
				datasetTypePath = path.join(deploymentPath, "test", "labels")
				!fs.existsSync(datasetTypePath) &&
					fs.mkdirSync(datasetTypePath, { recursive: true })
				// copy ( train/images, val/images, test/images)
				const promises = fileList.map((fm) => {
					return new Promise((resolve, reject) => {
						try {
							let destinationPath = path.join(
								deploymentPath,
								fm.DATASET_TYPE,
								"images",
								fm.FILE_NAME + fm.FILE_EXT
							)
							fs.copyFileSync(fm.FILE_PATH, destinationPath)

							// get image resolution
							const imageInfo = sizeOf(fm.FILE_PATH)
							// get polygonData
							const polygonData = JSON.parse(fs.readFileSync(fm.ANNO_DATA))

							destinationPath = path.join(
								deploymentPath,
								fm.DATASET_TYPE,
								"labels",
								fm.FILE_NAME + ".txt"
							)
							let annoData = ""
							polygonData.POLYGON_DATA.map((ppm) => {
								// mapping category number by name
								const categoryId = tagNames.indexOf(ppm.TAG_NAME)
								// change position to percent value
								// from x1,y1,x2,y2 to x(center) y(center) width height in 0-1 scale
								const anno = normalizeCoordinates(
									ppm.POSITION[0].X,
									ppm.POSITION[0].Y,
									ppm.POSITION[1].X,
									ppm.POSITION[1].Y,
									imageInfo.width,
									imageInfo.height
								)
								annoData += `${categoryId} ${anno.x} ${anno.y} ${anno.width} ${anno.height}\n`
							})
							logger.info(`Create annotation file: ${destinationPath}`)

							fs.writeFileSync(destinationPath, annoData, {
								encoding: "utf8",
								flag: "w",
							})

							resolve()
						} catch (e) {
							reject(e)
						}
					})
				})

				const results = await Promise.all(promises)
			}

			res.json({ result: true })
		} catch (e) {
			res.json({ result: e })
		}
		option.queryId = "updateDataSetStatus"
		option.param.DATASET_CD = datasetCd
		option.param.DATASET_STS = "DONE"
		await DH.executeQuery(option)
	})
)

router.post(
	"/removeDeployedDataSet",
	asyncHandler(async (req, res, next) => {
		try {
			let deploymentPath = path.join(
				config.deploymentPath,
				req.body.datasetName
			)
			fs.existsSync(deploymentPath) && rimraf.sync(deploymentPath)
			res.json({ result: true })
		} catch (e) {
			res.json({ result: e })
		}
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
			OBJECT_TYPE: el.OBJECT_TYPE,
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
			OBJECT_TYPE: el.OBJECT_TYPE,
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
		if (el.ANNO_DATA !== null && el.ANNO_DATA !== "") {
			if (path.extname(el.ANNO_DATA) === ".dat") {
				el.ANNO_DATA = fs.readFileSync(el.ANNO_DATA)
				let tempAnno = JSON.parse(el.ANNO_DATA)
				let tags = countUnique(tempAnno.POLYGON_DATA.map((ele) => ele.TAG_NAME))
				Object.keys(tags).map((key) => {
					tagInfo[key] = tagInfo[key] ? tagInfo[key] + tags[key] : tags[key]
				})
			}
		}
	})
	res.json(tagInfo)
})

module.exports = router
