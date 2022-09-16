"use strict"

var fs = require("fs")
var util = require("util")

class ConfigurationLoader {
  constructor() {
    this.config = {}
    this.instance = this
    this.configFiles = []
  }

  readConfigFile(filePath) {
    this.configFiles.push(filePath)

    var data = fs.readFileSync(filePath)
    if (data.length == 0) {
      throw new Error(
        "Error while reading/parsing config file [" + filePath + "]"
      )
    }

    var jsonData = JSON.parse(data)
    this.config = Object.assign(this.config, jsonData)
    return
  }

  get(property) {
    return this.config[property]
  }

  getConfig() {
    return this.config
  }

  reload() {
    this.config = {}
    for (var ind in this.configFiles) {
      this.readConfigFile(this.configFiles[ind])
    }
    return
  }

  toString() {
    return util.inspect(this.config, {
      showHidden: false,
      depth: null,
      colors: true
    })
  }

  clearValue(property, member) {
    if (member == null) {
      this.config[property] = undefined
    } else {
      this.config[property][member] = undefined
    }
  }

  checkValue() {
    var self = this

    var db = self.get("database")
    if (!db) {
      throw new Error("database is not set in the config file")
    }
    if (!db.address) {
      throw new Error("database.address is not set in the config file")
    }
    if (!db.port) {
      throw new Error("database.port is not set in the config file")
    }
    if (!db.dbname) {
      throw new Error("database.dbname is not set in the config file")
    }
    if (!db.user) {
      throw new Error("database.user is not set in the config file")
    }
    if (!db.password) {
      throw new Error("database.password is not set in the config file")
    }

    var config = self.getConfig()
    // Log
    !fs.existsSync(config.Log.dirPath) &&
      fs.mkdirSync(config.Log.dirPath, { recursive: true })
    !fs.existsSync(config.tempPath) &&
      fs.mkdirSync(config.tempPath, { recursive: true })
    !fs.existsSync(config.datasetPath) &&
      fs.mkdirSync(config.datasetPath, { recursive: true })
    !fs.existsSync(config.aiPath) &&
      fs.mkdirSync(config.aiPath, { recursive: true })
    // !fs.existsSync(config.trainingPath) && fs.mkdirSync(config.trainingPath)
    // !fs.existsSync(config.cutSampleRoot) && fs.mkdirSync(config.cutSampleRoot)

    // QI_SET OPTION
    config = config.QI_SET
    !fs.existsSync(config.isPath) &&
      fs.mkdirSync(config.isPath, { recursive: true })
    !fs.existsSync(config.prjPath) &&
      fs.mkdirSync(config.prjPath, { recursive: true })
  }
}

module.exports = new ConfigurationLoader()
