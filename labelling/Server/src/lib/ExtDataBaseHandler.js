import "@babel/polyfill"
import CL from "./ConfigurationLoader"

const config = CL.getConfig()

const logger = require("../lib/Logger")(__filename)

// "postinstall": "ln -s $HOME/Downloads/instantclient_19_8/libclntsh.dylib $(npm root)/oracledb/build/Release"

exports.extDBConnection = async (dbinfo) => {
  return new Promise((resolve, reject) => {
    try {
      const queryTimeoutMillsec =
        dbinfo.TIMEOUT === undefined ? 50000 : Number(dbinfo.TIMEOUT)
      const queryLimit = dbinfo.LIMIT === undefined ? 10 : Number(dbinfo.LIMIT)
      const config = {
        client: dbinfo.CLIENT,

        connection: {
          host: dbinfo.ADDRESS,
          port: dbinfo.PORT,
          user: dbinfo.USER,
          password: dbinfo.PASSWORD,
          database: dbinfo.DBNAME
        }
      }

      const knex = require("knex")
      const dbClient = knex(config)

      console.time("DB_TIME")
      dbinfo.include_limit = false
      if (dbinfo.include_limit)
        dbClient
          .raw(dbinfo.QUERY)
          .timeout(queryTimeoutMillsec)
          .then((data) => {
            resolve({ STATUS: 1, DATA: data })
          })
          .catch((err) => {
            console.log(err)
            resolve({ STATUS: 0, MSG: err })
          })
      else
        dbClient
          .with("with_alias", dbClient.raw(dbinfo.QUERY))
          .select("*")
          .from("with_alias")
          .limit(queryLimit)
          .timeout(queryTimeoutMillsec)
          .then((data) => {
            resolve({ STATUS: 1, DATA: data })
          })
          .catch((err) => {
            logger.error(`[exDB] query Fail \n${err}`)
            resolve({ STATUS: 0, MSG: err })
          })
      console.timeEnd("DB_TIME")
    } catch (error) {
      logger.error(`[exDB] Connection Fail \n${error.stack}`)
      resolve({ STATUS: 0, MSG: error.stack })
    }
  })
}
