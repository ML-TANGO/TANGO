#!/usr/bin/env node

/**
 * Module dependencies.
 */

var os = require("os")
var CL = require("./lib/ConfigurationLoader")
var CC = require("./lib/CommonConstants")
var cluster = require("cluster") // 클러스터 모듈 로드
var workers = []

var configPath = ""
switch (process.env.NODE_ENV) {
  case "production":
    configPath = "/config/server.json"
    break
  case "master":
    configPath = "/config/MasterServer.json"
    break
  default:
    process.env.NODE_ENV = "development"
    configPath = "/config/devServer.json"
    break
}
global.NODE_CONFIG_PATH = configPath
global.P_BUILD = "EE"
global.SERVICE_AUTH_LIST = []

CL.readConfigFile(__dirname + configPath)

var logger = require("./lib/Logger")(__filename)

var DH = require("./lib/DatabaseHandler")
var SOC = require("./lib/SocketHandler")
var MCS = require("./lib/MasterConfigSetter")

cluster.schedulingPolicy = cluster.SCHED_RR

function checkProcess() {
  var option = {}
  var psList = require("ps-list")
  option.source = CC.MAPPER.BIN
  option.param = {}
  option.queryId = "getAIprj"
  DH.executeQuery(option).then((qResult) => {
    psList().then((data) => {
      data.map((ele) => {
        qResult.map((item) => {
          if (ele.pid == item.AI_PID) {
            qResult.pop(item)
          }
        })
      })

      if (qResult.length > 0) {
        option.queryId = "stopAiFirstCheck"
        option.param.DATA = qResult
        DH.executeQuery(option)
      }
    })
  })

  // var psList = require("ps-list")
  // psList().then((data) => {
  //   console.log(data)
  // })
}

if (cluster.isMaster) {
  DH.init()
  var SH = require("./lib/SchedulerHandler")
  logger.info(
    `Server Initializing (1/5) Mode [\x1b[33m${process.env.NODE_ENV}\x1b[0m]`
  )

  //소켓 생성
  SOC.init(3000)
  logger.info(
    `Server Initializing (2/5) Realtime Socket port open [\x1b[33m${CL.get(
      "soc_port"
    )}\x1b[0m]`
  )

  checkProcess()
  //스케쥴러 초기화
  // SH.startCheckJob()
  // SH.startGpuCheckJob()

  logger.info(
    `Server Initializing (3/5) Schedule List [\x1b[33m${SH.getScheduleList()}\x1b[0m]`
  )

  if (MCS.setMasterConfig(NODE_CONFIG_PATH))
    logger.info(
      `Server Initializing (4/5) Master Config Setting [\x1b[33mSUCCESS\x1b[0m]`
    )
  else
    logger.info(
      `Server Initializing (4/5) Master Config Setting [\x1b[31mFAIL\x1b[0m]`
    )

  // MCS.setNvidiaSet()
  MCS.initActiveProcess()

  let workerSize = os.cpus().length / 2

  if (workerSize > 8) workerSize = 8
  logger.info(
    `Server Initializing (5/5) Create [\x1b[33m${workerSize}\x1b[0m]Workers`
  )
  // 마스터 처리
  for (var i = 0; i < workerSize; i++) {
    let worker = cluster.fork() // CPU 개수만큼 fork
    logger.info(
      `---------------------WorkerId:\x1b[33m${worker.id}\x1b[0m, PID:\x1b[33m${worker.process.pid}\x1b[0m`
    )
    worker.on("message", (msg) => {
      if (msg.isSoc !== undefined && msg.isSoc === 1) {
        SOC.setSocData(msg.data, msg.type)
      } else if (msg.isSECH !== undefined && msg.isSECH === 1) {
      } else if (msg.isBuild) {
        workers.map((ch) => {
          ch.send({
            cmd: "val_BUILD:edit",
            data: msg.build
          })
        })
      } else if (msg.isService !== undefined && msg.isService === "set") {
        SERVICE_AUTH_LIST.push(msg.data)
        workers.map((ch) => {
          ch.send({
            cmd: "val_SERVICE:edit",
            data: SERVICE_AUTH_LIST,
            addData: msg.data
          })
        })
      } else if (msg.isService !== undefined && msg.isService === "pop") {
        SERVICE_AUTH_LIST = msg.listdata
        workers.map((ch) => {
          ch.send({
            cmd: "val_SERVICE:edit",
            data: SERVICE_AUTH_LIST,
            addData: msg.data
          })
        })
      }
    })
    workers.push(worker)
  }
  logger.info(
    `[\x1b[33m${
      process.env.NODE_ENV
    }\x1b[0m]Server running on [\x1b[33m${CL.get("ip")}:${CL.get(
      "port"
    )}\x1b[0m]  `
  )

  // 워커 종료시 다잉 메시지 출력
  cluster.on("exit", function (worker, code, signal) {
    if (code === 200) cluster.fork()
    // console.log("worker " + worker.process.pid + " died")
  })

  process.on("SIGTERM", async () => {
    SOC.close()
    logger.info("[SYS] Server STOP")
    process.exit(1)
  })

  process.on("SIGINT", async () => {
    SOC.close()
    logger.error("[SYS] [SYS] Server STOP by CTRL+C")
    process.exit(1)
  })
} else {
  // 워커 처리
  var app = require("./app")
  var debug = require("debug")("BluAI:server")
  var http = require("http")

  process.on("message", (message) => {
    if (message.cmd === "val_BUILD:edit") {
      P_BUILD = message.data
      logger.info(`[SYSTEM] Set Build Version ${P_BUILD}`)
      // console.log(
      //   `[${process.pid}:globalData:edit] ${JSON.stringify(globalData)}`
      // )
    }

    if (message.cmd === "val_SERVICE:edit") {
      SERVICE_AUTH_LIST = message.data
      logger.info(`[SYSTEM] Set Service IS_CD: [${message.addData}]`)
      // console.log(
      //   `[${process.pid}:globalData:edit] ${JSON.stringify(globalData)}`
      // )
    }
  })
  /**
   * Get port from environment and store in Express.
   */

  var port = normalizePort(CL.get("port") || "3001")

  app.set("port", port)
  /**
   * Create HTTP server.
   */
  // SOC.init(3000)
  //
  var server = http.createServer(app)

  /**
   * Listen on provided port, on all network interfaces.
   */

  server.listen(port)

  //server.on("error", onError)
  server.on("listening", onListening)

  //logger.get("app").info("Server listening on %s port", port)
  /**
   * Normalize a port into a number, string, or false.
   */

  function normalizePort(val) {
    var port = parseInt(val, 10)

    if (isNaN(port)) {
      // named pipe
      return val
    }

    if (port >= 0) {
      // port number
      return port
    }

    return false
  }

  /**
   * Event listener for HTTP server "error" event.
   */

  function onError(error) {
    if (error.syscall !== "listen") {
      throw error
    }

    var bind = typeof port === "string" ? "Pipe " + port : "Port " + port
    // handle specific listen errors with friendly messages
    switch (error.code) {
      case "EACCES":
        console.error(bind + " requires elevated privileges")
        process.exit(1)
        break
      case "EADDRINUSE":
        console.error(bind + " is already in use")
        process.exit(1)
        break
      default:
        throw error
    }
  }

  /**
   * Event listener for HTTP server "listening" event.
   */

  function onListening() {
    var addr = server.address()
    var bind = typeof addr === "string" ? "pipe " + addr : "port " + addr.port
    debug("Listening on " + bind)
  }
}
