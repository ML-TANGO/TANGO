import express from "express"
import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
import passport from "passport"
import {
	register,
	changepw,
	login,
	refreshToken,
	logout
} from "../lib/Authentication"

var router = express.Router()

router.get("/start", async (req, res, next) => {
	try {
		const prjId = req.query.project_id
		const userId = req.query.user_id

		let option = {}
		option.source = CC.MAPPER.TANGO
		option.queryId = "setStart"
		option.param = {
			PROJECT_ID : prjId,
			USER_ID : userId,
		}
		await DH.executeQuery(option)
		res.set("Content-Type", "text/plain")
		res.status(200).send("starting")
	} catch (error) {
		res.status(200).send("error\n"+error)
	}
})

router.get("/stop", async (req, res, next) => {
		try {
			const prjId = req.query.project_id
			const userId = req.query.user_id
	
			let option = {}
			option.source = CC.MAPPER.TANGO
			option.queryId = "deleteProject"
			option.param = {
				PROJECT_ID : prjId,
				USER_ID : userId,
			}
			let list = await DH.executeQuery(option)
			console.log(list[0])
	
			res.set("Content-Type", "text/plain")
			res.status(200).send("finished")
		} catch (error) {
			res.status(200).send("error\n"+error)
		}
})


router.get("/test", async (req, res, next) => {
	console.log("!!!!!!!")
	let option = {}
			option.source = CC.MAPPER.TANGO
			option.queryId = "getProjectInfo"
			let list = await DH.executeQuery(option)
			console.log(list[0])
			res.set("Content-Type", "text/plain")
			res.status(200).send("finished")
})

export default router
