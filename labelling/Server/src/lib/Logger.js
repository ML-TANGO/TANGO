const winston = require("winston")
require("winston-daily-rotate-file")
const CL = require("./ConfigurationLoader")

const logger = function (module) {
  let logInfo = CL.get("Log") || {}
  let logLevel = logInfo.level || "debug"
  let logFormat = winston.format.printf(function ({
    level,
    message,
    timestamp,
    label
  }) {
    // return `[${timestamp}][${level}]:[${label}] = ${message}`
    return `[${timestamp}][${level}] = ${message}`
  })

  let transports = [
    new winston.transports.Console({
      format: winston.format.combine(winston.format.colorize(), logFormat)
    })
  ]
  if (logInfo.save === "true") {
    transports.push(
      new winston.transports.DailyRotateFile({
        filename: "bluai-server-%DATE%.log",
        dirname: logInfo.dirPath || "./",
        datePattern: "YYYY-MM-DD",
        zippedArchive: true,
        maxSize: "20m",
        maxFiles: "14d"
      })
    )
  }

  return winston.createLogger({
    level: logLevel,
    format: winston.format.combine(
      winston.format(function (info) {
        info.level = info.level.toUpperCase().padEnd(5)
        return info
      })(),
      winston.format.splat(),
      winston.format.timestamp({
        format: "YYYY-MM-DD HH:mm:ss"
      }),
      winston.format.label({
        label: module != undefined ? module.split("/").pop().padEnd(20) : ""
      }),
      logFormat
    ),
    transports: transports,

    exitOnError: false
  })
}

module.exports = logger
