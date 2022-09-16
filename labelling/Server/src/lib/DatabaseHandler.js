import mybatisMapper from "mybatis-mapper"
import path from "path"
import mysql from "mysql"

import CL from "./ConfigurationLoader"
var logger = require("../lib/Logger")(__filename)

let mapperPath = ""

class DatabaseHandler {
  constructor() {
    this.client = {}
    this.instance = this
  }
  init() {
    mapperPath = path.dirname(path.dirname(__filename))
    mapperPath = path.join(mapperPath, "routes/sqlMapper/")
    //https://nesoy.github.io/articles/2017-04/Nodejs-MySQL
    const dbinfo = CL.get("database")
    const config = {
      host: dbinfo["address"],
      user: dbinfo["user"],
      password: dbinfo["password"],
      port: dbinfo["port"],
      database: dbinfo["dbname"],
      connectionLimit: dbinfo["maxConnection"],
      charset: "utf8", // 이 부분을 추가하면 된다.
      acquireTimeout: 0
    }
    this.client = mysql.createPool(config)
  }

  executeQuery(option) {
    const format = { language: "sql", indent: "  " }
    const source = option.source + ".xml"
    const nameSpace = option.source
    const queryId = option.queryId
    const param = option.param


    mybatisMapper.createMapper([mapperPath + source])
    try {
      var query = mybatisMapper.getStatement(nameSpace, queryId, param, format)

      if (queryId === "setDBInfo") logger.info(`[!!!] setDBInfo \n${query}`)
      // console.log("======[" + queryId + "]=========")
      // console.log(query)
      // console.log("================")
      return new Promise((resolve, reject) => {
        this.client.getConnection((connError, conn) => {
          if (connError) {
            logger.error("[DB] Connection Error")
            reject(connError)
          } else {
            conn.query(query, (sqlError, data) => {
              if (sqlError) {
                logger.error(`"[DB] Connection Error [${queryId}]"`)
                logger.error(query)
                logger.error(sqlError.message)
                reject(sqlError)
              } else {
                resolve(data)
              }
            })
          }
          try {
            conn.release()
          } catch (error) {
            console.log("db error")
          }
        })
      })
    } catch (queryBindError) {
      logger.error(`"[DB] Bind Error [${queryId}]" in [${source}]`)
      logger.error(queryBindError.message)
      return Promise.reject(queryBindError.message)
    }
  }
}

module.exports = new DatabaseHandler()
