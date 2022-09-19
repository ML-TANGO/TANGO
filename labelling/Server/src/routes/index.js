import express from "express"
import passport from "passport"

import DataSet from "./DataSetController"
import AiProject from "./AiProjectController"
import ImageAnno from "./ImageAnnoController"
import VideoAnno from "./VideoAnnoController"
import Binary from "./BinaryController"
import System from "./SystemController"
import Auth from "./AuthController"
// import Eval from "./EvaluationController"

//Tabular
import TB_DataSet from "./Tab/TabDataSetController"
import TB_AiPrj from "./Tab/TabAiProjectController"
import TB_Bin from "./Tab/TabBinaryController"

const router = express.Router()

router.use("/*", (req, res, next) => {
  res.setHeader("Expires", "-1")
  res.setHeader("Cache-Control", "must-revalidate, private")
  next()
})

router.use("/auth", Auth)
router.use(
  "/dataset",
  // passport.authenticate("jwt", { session: false }),
  DataSet
)
router.use(
  "/aiproject",
  // passport.authenticate("jwt", { session: false }),
  AiProject
)
router.use(
  "/imageanno",
  // passport.authenticate("jwt", { session: false }),
  ImageAnno
)
router.use(
  "/videoanno",
  // passport.authenticate("jwt", { session: false }),
  VideoAnno
)
router.use("/binary", Binary)
router.use(
  "/system",
  // passport.authenticate("jwt", { session: false }),
  System
)

router.use(
  "/tab/dataset",
  // passport.authenticate("jwt", { session: false }),
  TB_DataSet
)

router.use(
  "/tab/aiproject",
  // passport.authenticate("jwt", { session: false }),
  TB_AiPrj
)

router.use(
  "/tab/binary",
  // passport.authenticate("jwt", { session: false }),
  TB_Bin
)

// router.use(
//   "/service",
//   // passport.authenticate("jwt", { session: false }),
//   Service
// )

// router.use(
//   "/eval",
//   // passport.authenticate("jwt", { session: false }),
//   Eval
// )

module.exports = router
