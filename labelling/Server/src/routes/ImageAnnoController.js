import express from "express"
import asyncHandler from "express-async-handler"
import path from "path"
import fs from "fs"

import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
import CL from "../lib/ConfigurationLoader"
import CF from "../lib/CommonFunction"

const logger = require("../lib/Logger")(__filename)
const router = express.Router()
const config = CL.getConfig()

const _base64Decode = (base64, path) => {
	fs.writeFileSync(path, new Buffer.from(base64), "base64")
}

router.post(
	"/getImageList",
	asyncHandler(async (req, res, next) => {
		let thumUrl = "/static/"

		let option = {}
		option.source = CC.MAPPER.IMG_ANNO
		option.queryId = "getImageList"
		option.param = req.body
		option.param.DATA_STATUS = "ORG"

		thumUrl += option.param.DATASET_CD + "/"

		let list = await DH.executeQuery(option)
		list.map((ele) => {
			let FILE_PATH
			const temp = ele.FILE_PATH.indexOf(option.param.DATASET_CD)
			if (temp > -1)
				FILE_PATH = ele.FILE_PATH.substr(
					option.param.DATASET_CD.length + temp,
					ele.FILE_PATH.length
				)

			let thumPath = thumUrl + FILE_PATH
			thumPath = path.join(
				path.dirname(thumPath),
				"THUM_" + path.basename(thumPath)
			)
			ele.THUM = thumPath
			ele.FILE_SIZE = ele.FILE_SIZE
			ele.IMG_WITH = "..."
			ele.IMG_HEIGHT = "..."
			ele.IMG_INFO = "..."
		})
		res.json(list)
	})
)

router.post(
	"/getImage",
	asyncHandler(async (req, res, next) => {
		let isTest = req.body.IS_TEST
		if (isTest === undefined || isTest === null) isTest = false
		let option = {}
		option.source = CC.MAPPER.IMG_ANNO
		option.queryId = "getImage"
		option.param = req.body

		let imgFile = await DH.executeQuery(option)

		option.queryId = "getDataTags"
		let tagList = await DH.executeQuery(option)

		imgFile = imgFile[0]
		if (imgFile.ANNO_DATA !== null && imgFile.ANNO_DATA !== "" && !isTest) {
			if (req.body.OBJECT_TYPE === "S" || req.body.OBJECT_TYPE === "D") {
				if (path.extname(imgFile.ANNO_DATA) === ".dat") {
					try {
						imgFile.ANNO_DATA = fs.readFileSync(imgFile.ANNO_DATA)
						let tempAnno = JSON.parse(imgFile.ANNO_DATA)
						tempAnno.POLYGON_DATA.map((ele) => {
							// jogoon 수정, TAG_CD가 없을때 에러. undefined check 추가
							if (ele.TAG_CD != undefined && ele.TAG_CD == null) {
								let findTag = tagList.find((item) => item.COLOR === ele.COLOR)
								ele.TAG_CD = findTag.TAG_CD
								ele.TAG_NAME = findTag.NAME
							}
						})
						imgFile.ANNO_DATA = tempAnno
					} catch (e) {
						logger.error(e)
					}
				}
			} else {
				// 이전 로직.
				imgFile.ANNO_DATA = imgFile.ANNO_DATA.substr(
					0,
					imgFile.ANNO_DATA.length - 1
				)
				const rects = imgFile.ANNO_DATA.split(";")

				imgFile.ANNO_DATA = { POLYGON_DATA: [], BRUSH_DATA: [] }
				rects.map((ele) => {
					let temp = ele.split(",")
					imgFile.ANNO_DATA.POLYGON_DATA.push({
						DATA_CD: imgFile.DATA_CD,
						DATASET_CD: imgFile.DATASET_CD,
						POSITION: [
							{ X: temp[0], Y: temp[1] },
							{ X: temp[2], Y: temp[3] },
						],
						TAG_CD: temp[4],
						COLOR: temp[5],
					})
				})
				option.param.DATA = imgFile.ANNO_DATA.POLYGON_DATA
				option.param.DATASET_CD = imgFile.DATASET_CD
				option.queryId = "getDataTagInfoByTags"
				let inTagInfo = await DH.executeQuery(option)
				if (inTagInfo.length !== 0) {
					imgFile.ANNO_DATA.POLYGON_DATA.map((anno) => {
						anno.TAG_NAME = inTagInfo.filter(
							(v) => String(v.TAG_CD) === String(anno.TAG_CD)
						)[0]?.NAME
					})
				}
			}
		} else imgFile.ANNO_DATA = { POLYGON_DATA: [], BRUSH_DATA: [] }

		let url = "/static"
		url = path.join(url, imgFile.DATASET_CD)
		url = path.join(url, imgFile.FILE_RELPATH)
		imgFile.FILE_URL = url
		res.json(imgFile)
	})
)

router.post(
	"/getDataTags",
	asyncHandler(async (req, res, next) => {
		let option = {}
		option.source = CC.MAPPER.IMG_ANNO
		option.param = req.body
		option.queryId = "getDataSet"
		let datasetInfo = await DH.executeQuery(option)
		datasetInfo = datasetInfo[0]
		option.param.AI_CD = datasetInfo.AUTO_MODEL

		option.queryId = "getDataTags"

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
									value: item.CATEGORY_SEQ,
								})
							}
							if (item.DEPTH === 1 && item.OBJECT_TYPE === objType) {
								ele.CATEGORY2_LIST.push({
									label: item.CATEGORY_NAME,
									value: item.CATEGORY_SEQ,
								})
							}
							if (item.DEPTH === 2 && item.OBJECT_TYPE === objType) {
								ele.CATEGORY3_LIST.push({
									label: `${item.CLASS_DP_NAME} (${item.CATEGORY_NAME}) - ${item.BASE_MDL}`,
									value: item.CLASS_CD,
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

		const result = await Promise.all(promises)
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
	"/updateDataTag",
	asyncHandler(async (req, res, next) => {
		let option = {}
		option.source = CC.MAPPER.IMG_ANNO
		option.queryId = "updateDataTag"
		option.param = req.body
		await DH.executeQuery(option)

		option.queryId = "getImageList"
		option.param.DATA_STATUS = "ORG"

		let list = await DH.executeQuery(option)

		const chFile = list.map((ele) => {
			return new Promise((resolve, reject) => {
				let filePath = path.join(
					path.parse(ele.FILE_PATH).dir,
					path.parse(ele.FILE_PATH).name
				)
				filePath += ".dat"
				if (fs.existsSync(filePath)) {
					var rsFile = JSON.parse(String(fs.readFileSync(filePath)))
					rsFile.POLYGON_DATA.map((frame) => {
						if (req.body.TAG_CD === frame.TAG_CD) {
							frame.TAG_NAME = req.body.NAME
							frame.COLOR = req.body.COLOR
						}
					})
					fs.writeFileSync(filePath, JSON.stringify(rsFile), {
						encoding: "utf8",
						flag: "w",
					})
				}
				resolve(1)
			})
		})

		await Promise.all(chFile)

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
	"/testFunc",
	asyncHandler(async (req, res, next) => {
		CF.checkModel("", "", -1)
	})
)

router.post(
	"/getImagePredict",
	asyncHandler(async (req, res, next) => {
		const isReady = req.body.IS_READY
		let activeCheck = await CF.checkModel(req.body.AI_CD, isReady, -1)
		let result = {}
		let runPredict = activeCheck.runPrc
		let returnCode = activeCheck.code

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
					},
				],
				OBJECT_TYPE: option.param.OBJECT_TYPE,
				CLASS_DB_NM: classInfo.CLASS_DB_NAME,
				DATA_TYPE: "I",
				AI_TYPE: "PRE",
				AI_CD: classInfo.BASE_MDL,
				CALSS_CD: classInfo.CALSS_CD,
				MDL_PATH: classInfo.MDL_PATH,
				EPOCH: -1,
				IS_TEST: false,
			}
			try {
				let mdlPid = await CF.sendRequestResLong(
					config.masterIp,
					config.masterPort,
					CC.URI.modelLoad,
					param
				)
				if (mdlPid.PID === 0) throw new Error("Model Load Fail")
				if (isReady)
					result = await CF.sendRequestResLong(
						config.masterIp,
						config.masterPort,
						CC.URI.miniPredictor,
						param
					)
				else result = { status: 1 }
			} catch (error) {
				result = { status: 0, msg: error }
			} finally {
				res.json(result)
			}
		} else res.json(returnCode)
	})
)

router.post(
	"/stopActivePredictor",
	asyncHandler(async (req, res, next) => {
		let result = {}
		try {
			result = await CF.sendRequestResLong(
				config.masterIp,
				config.masterPort,
				CC.URI.activePredictor,
				{}
			)
		} catch (error) {}
	})
)

router.post(
	"/getActivePredictor",
	asyncHandler(async (req, res, next) => {
		let result = {}
		try {
			result = await CF.sendRequestResLong(
				config.masterIp,
				config.masterPort,
				CC.URI.activePredictor,
				{}
			)

			if (result.length >= 0) {
				const dbPromises = result.map((item) => {
					return new Promise((resolve, reject) => {
						let option = {}
						option.source = CC.MAPPER.IMG_ANNO
						option.queryId = "getActiveClass"
						option.param = item
						let ele = DH.executeQuery(option)
						resolve(ele)
					})
				})

				await Promise.all(dbPromises).then((data) => {
					result.map((ele, idx) => {
						ele.USEABLE_CLASS = data[idx]
					})
				})
			}
		} catch (error) {
			logger.error(error)
			result = { status: 0, msg: error }
		} finally {
			res.json(result)
		}
	})
)
router.post(
	"/setImageAnnotation",
	asyncHandler(async (req, res, next) => {
		const param = req.body.ANNO_DATA
		const type = req.body.OBJECT_TYPE
		let option = {}
		option.source = CC.MAPPER.IMG_ANNO
		option.param = {}
		option.param.DATASET_CD = req.body.DATASET_CD
		option.param.DATA_CD = req.body.DATA_CD

		if (type === "D") {
			const orgFile = req.body.FILE_PATH
			const dataFile = path.join(
				path.dirname(orgFile),
				path.basename(orgFile, path.extname(orgFile)) + ".dat"
			)
			fs.writeFileSync(dataFile, JSON.stringify(param))
			let tagCds = []
			param.POLYGON_DATA.map((ele) => {
				tagCds.push(String(ele.TAG_CD))
			})
			const unique = Array.from(new Set(tagCds))

			option.param.TAG_CNT = unique.length
			option.param.ANNO_CNT = param.POLYGON_DATA.length
			option.param.IS_ANNO = true
			option.param.ANNO_DATA = dataFile
			option.queryId = "setUpdateDataElement"

			await DH.executeQuery(option)
		} else if (type === "S") {
			//세그멘테이션의 경우 그리기 자취 파일로저장, 마스크 파일 저장
			const orgFile = req.body.FILE_PATH
			const dataFile = path.join(
				path.dirname(orgFile),
				path.basename(orgFile, path.extname(orgFile)) + ".dat"
			)
			const maskFile = path.join(
				path.dirname(orgFile),
				"MASK_" + path.basename(orgFile, path.extname(orgFile)) + ".png"
			)
			const mask = req.body.MASK_IMG

			_base64Decode(mask, maskFile)
			fs.writeFileSync(dataFile, JSON.stringify(param))
			let tagCds = []
			param.POLYGON_DATA.map((ele) => {
				tagCds.push(String(ele.TAG_CD))
			})
			const unique = Array.from(new Set(tagCds))

			option.param.TAG_CNT = unique.length
			option.param.ANNO_CNT = param.POLYGON_DATA.length
			option.param.IS_ANNO = true
			option.param.ANNO_DATA = dataFile
			option.queryId = "setUpdateDataElement"

			await DH.executeQuery(option)
		}
		res.json({ status: 1 })
	})
)

module.exports = router
