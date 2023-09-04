import "@babel/polyfill"
import createError from "http-errors"
import express from "express"
import morgan from "morgan"
import bodyParser from "body-parser"

import indexRouter from "./routes/index"
import tangoRouter from "./routes/tangoStatus"
import CL from "./lib/ConfigurationLoader"
import DH from "./lib/DatabaseHandler"
import auth from "./routes/auth"
import { authInit } from "./lib/Authentication"

var app = express()
var cors = require("cors")()
var logger = require("./lib/Logger")(__filename)

app.use(cors)
app.use(function (req, res, next) {
	// req.setTimeout(60 * 10 * 1000) // set timeout 3 min.
	res.setTimeout(60 * 10 * 1000, function () {
		// res.setTimeout(10 * 1000, function () {
		console.log("Request has timed out.")
		res.send(408)
	})
	res.header("Access-Control-Allow-Origin", "*")
	res.header("Access-Control-Allow-Headers", "X-Requested-With")
	res.header("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE")
	next()
})
//리퀘스트를 위한 파서 설정
//app.use(logger("dev"))
app.use(bodyParser.json({ limit: "5000mb" }))
app.use(bodyParser.urlencoded({ extended: false }))
app.use(
	morgan("dev", {
		stream: {
			write: function (message) {
				logger.debug(message.replace("\n", ""))
			},
		},
	})
)

CL.checkValue()

/**
 * Route Setting
 */
app.use("/static", express.static(CL.getConfig().datasetPath))
app.use("/qithum", express.static(CL.getConfig().QI_SET.isPath))
// app.use("/api/auth", auth)
app.use("/api", indexRouter)
app.use("/", tangoRouter)

// SOC.init(CL.get("soc_port"))

//데이터베이스 핸들러 초기화
DH.init()
// catch 404 and forward to error handler
app.use(function (req, res, next) {
	next(createError(404))
})

authInit()

// error handler
app.use(function (err, req, res, next) {
	switch (err.code) {
		case "ER_DUP_ENTRY":
			res.status(200).json({ status: 0, code: "sql_dup", msg: err.message })
			break
		case "Bin":
			err = JSON.stringify(err)
			logger.error(`[SYS] Binary Error\n ${String(err)}`)
			res.status(400).json({ status: 0, code: "Bin_Error", msg: String(err) })
			break
		case undefined:
			logger.error(err.stack)
			res
				.status(500)
				.json({ status: 0, code: "Undefined_Error", msg: String(err) })
			break

		default:
			res.status(400).json({ status: 0, code: "Error", msg: err.message })
			logger.error(`default \n ${String(err)}`)
			next()
			break
	}
})

process.on("uncaughtException", async (error) => {
	if (error.syscall !== "listen") {
		throw error
	}

	var bind = CL.get("port")

	// handle specific listen errors with friendly messages
	switch (error.code) {
		case "EACCES":
			logger.error(
				`[SYS] ${bind} requires elevated privileges \n ${error.stack}`
			)
			process.exit(1)
			break
		case "EADDRINUSE":
			logger.error(`[SYS] ${bind} is already in use \n ${error.stack}`)
			process.exit(1)
			break
		default:
			logger.error(`[SYS] Server STOP by Uncaught Exception \n ${error.stack}`)
			process.exit(1)
			break
	}
})

module.exports = app
