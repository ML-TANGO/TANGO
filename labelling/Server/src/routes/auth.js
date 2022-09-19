import express from "express"
import passport from "passport"
import {
	register,
	changepw,
	login,
	refreshToken,
	logout
} from "../lib/Authentication"

var router = express.Router()

router.post("/login", login)
router.post("/refresh", refreshToken)
router.post("/logout", passport.authenticate("jwt", { session: false }), logout)
router.post(
	"/register",
	passport.authenticate("jwt", { session: false }),
	register
)
router.post(
	"/chagepw",
	passport.authenticate("jwt", { session: false }),
	changepw
)

export default router
