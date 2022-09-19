import express from "express"
import DH from "../lib/DatabaseHandler"
import CC from "../lib/CommonConstants"
var router = express.Router()

router.post("/:type", async function(req, res, next) {
  const codeType = req.params.type

  let option = {}

  let codePlant = []
  let codeStatus = []
  let codeMeasurementType = []
  let codeMaker = []
  let codePurpose = []
  let codeCustomer = []
  let auth = []

  option.source = CC.MAPPER_NAME.CODETABLE
  option.nameSpace = "CodeTableMapper"
  option.queryId = "getCodeTable"

  await DH.executeQuery(option)
    .then(data => {
      if(data != undefined)
        data.map(item => {
          let code = {}
          switch (item.CD_KD) {
            case '001':
                code.value=item.CD_NM_DB
                code.label=item.CD_NM_DP
                codeCustomer.push(code)
              break
            case '002':
                code.value=item.CD_NM_DB
                code.label=item.CD_NM_DP
                codeMaker.push(code)
              break
            case '003':
                code.value=item.CD_NM_DB
                code.label=item.CD_NM_DP
                codePlant.push(code)
              break
            case '004':
                code.value=item.CD_NM_DB
                code.label=item.CD_NM_DP
                codePurpose.push(code)
              break
            case '005':
                code.value=item.CD_NM_DB
                code.label=item.CD_NM_DP
                codeMeasurementType.push(code)
              break
            case '006':
                code.value=item.CD_NM_DB
                code.label=item.CD_NM_DP
                codeStatus.push(code)
              break
            case '008':
                code.value=item.CD_NM_DB
                code.label=item.CD_NM_DP
                auth.push(code)
              break
          }
        })
    })
    .catch(err => {
      res.status(500).send({
        error: "DataBase Error",
        msg: String(err.message)
      })
    })

  switch (codeType) {
    case "plant":
      res.json(codePlant)
      break
    case "status":
      res.json(codeStatus)
      break

    case "measurementType":
      res.json(codeMeasurementType)
      break

    case "customer":
      res.json(codeCustomer)
      break

    case "maker":
      res.json(codeMaker)
      break

    case "purpose":
      res.json(codePurpose)
      break
    case "auth":
      res.json(auth)
      break
  }
})
module.exports = router
