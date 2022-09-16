import passport from "passport"
import passportJWT from "passport-jwt"
import jwt from "jsonwebtoken"
import crypto from "crypto"
import DH from "./DatabaseHandler"
import CL from "./ConfigurationLoader"
import os from "os"
import CC from "./CommonConstants"

const LocalStrategy = require("passport-local").Strategy
const secretKey = "weda1!"
const salt = "Z3JlZW53aGFsZXM="

const JWTStrategy = passportJWT.Strategy
const ExtractJWT = passportJWT.ExtractJwt
const logger = require("../lib/Logger")(__filename)

var ssoTempToken = {}

const updateCount = async (userId, count) => {
  let option = {
    source: CC.MAPPER.AUTH,
    queryId: "getUsersAll",
    param: {
      USER_ID: userId
    }
  }

  const userInfo = await DH.executeQuery(option)
  if (userInfo.length !== 0) {
    const loginCnt =
      count !== undefined ? count : Number(userInfo[0].LOGIN_CNT) + 1
    option = {
      source: CC.MAPPER.AUTH,
      queryId: "updateUser",
      param: {
        USER_ID: userId,
        LOGIN_CNT: String(loginCnt)
      }
    }
    await DH.executeQuery(option)
    return loginCnt
  } else {
    return null
  }
}

export const authInit = () => {
  // Local Strategy
  passport.use(
    new LocalStrategy(
      {
        usernameField: "USER_ID",
        passwordField: "USER_PW"
      },
      function (USER_ID, USER_PW, done) {
        let option = {
          source: CC.MAPPER.AUTH,
          queryId: "getUsersAll",
          param: {
            USER_ID,
            USER_PW: crypto
              .createHash("sha512")
              .update(USER_PW + salt)
              .digest("hex")
          }
        }

        return DH.executeQuery(option)
          .then(async (user) => {
            /*
              1 : 로그인 성공
              2 : 아이디 or 암호 틀림
              3 : 패스워드 틀림 5회 이상 ID LOCK
            */
            if (user == undefined || user.length == 0) {
              // 비밀번호 틀렸을 경우 Count Update
              const loginCount = await updateCount(USER_ID)
              if (loginCount >= 5) {
                option = {
                  source: CC.MAPPER.AUTH,
                  queryId: "updateUser",
                  param: {
                    USER_ID: USER_ID,
                    USE: "2"
                  }
                }
                await DH.executeQuery(option)
                return done(null, false, {
                  STATUS: 3,
                  message: "Your ID is locked. Please contact the administrator"
                })
              } else {
                return done(null, false, {
                  STATUS: 2,
                  loginCount: loginCount,
                  message: "Incorrect ID or password."
                })
              }
            } else {
              // ID LOCK
              if (user[0].USE === "2") {
                return done(null, false, {
                  STATUS: 3,
                  message: "Your ID is locked. Please contact the administrator"
                })
              }

              // 비밀번호 변경 후 90일 이상 경과 체크
              // const diff = Math.floor(
              // 	moment.duration(moment().diff(moment(user[0].CREATED))).asDays()
              // )

              let retUserInfo = {
                USER_ID: user[0].USER_ID,
                USER_NM: user[0].USER_NM,
                ROLE: user[0].ROLE,
                USE: user[0].USE,
                // DIFF: diff >= 90 ? true : false,
                STATUS: 1
              }
              return done(null, retUserInfo, {
                message: "Logged In Successfully"
              })
            }
          })
          .catch((err) => done(err))
      }
    )
  )

  //JWT Strategy
  passport.use(
    new JWTStrategy(
      {
        jwtFromRequest: ExtractJWT.fromAuthHeaderAsBearerToken(),
        secretOrKey: secretKey
      },
      function (jwtPayload, done) {
        if (jwtPayload == undefined || jwtPayload.USER_ID == undefined) {
          done("token was broken.")
        } else {
          done(null, jwtPayload)
        }
      }
    )
  )
}
export const register = (req, res, next) => {
  const UserInfo = req.body

  let option = {
    source: CC.MAPPER.AUTH,
    queryId: "setUsers",
    param: {
      USER_ID: UserInfo.USER_ID,
      USER_NM: UserInfo.USER_NM,
      USER_PW: crypto
        .createHash("sha512")
        .update(UserInfo.USER_PW + salt)
        .digest("hex"),
      ROLE: UserInfo.ROLE,
      USE: UserInfo.USE,
      PREV_PW: crypto
        .createHash("sha512")
        .update(UserInfo.USER_PW + salt)
        .digest("hex")
    }
  }

  DH.executeQuery(option)
    .then((result) => {
      res.send({ status: 1, message: `Created USER : [${UserInfo.USER_ID}]` })
    })
    .catch((err) => {
      res.status(400).send(err)
      logger.info(`Failed to create user [${UserInfo.USER_ID}] : ${err}`)
    })
}

export const changepw = (req, res, next) => {
  const UserInfo = req.body
  if (UserInfo.USER_ID == undefined) {
    const logMsg = `USER_ID must not be empty`
    return next(new Error(logMsg))
  } else if (UserInfo.USER_PW == undefined) {
    const logMsg = `USER_PW must not be empty`
    return next(new Error(logMsg))
  }
  let option = {
    source: CC.MAPPER.AUTH,
    queryId: "getUsersAll",
    param: {
      USER_ID: UserInfo.USER_ID
    }
  }

  DH.executeQuery(option)
    .then((user) => {
      const pw = crypto
        .createHash("sha512")
        .update(UserInfo.USER_PW + salt)
        .digest("hex")
      if (user[0]?.USER_PW !== pw && user[0]?.PREV_PW !== pw) {
        UserInfo.USER_PW = pw
        UserInfo.PREV_PW = user[0]?.USER_PW
        UserInfo.LOGIN_CNT = 0
      }
      let option = {
        source: CC.MAPPER.AUTH,
        queryId: "updateUser",
        param: UserInfo
      }
      DH.executeQuery(option)
      res.send({
        status: 1,
        message: `Updated USER password [${UserInfo.USER_ID}]`
      })
    })
    .catch((err) => {
      next(err)
      logger.info(
        `Failed to update user password [${UserInfo.USER_ID}] : ${err}`
      )
    })
}

export const login = (req, res, next) => {
  passport.authenticate("local", { session: false }, (err, user, msg) => {
    if (err || !user) {
      return next(new Error(msg.message))
    }

    req.login(user, { session: false }, (err) => {
      if (err) {
        logger.error(err)
        return next(err)
      }
      const result = _createToken(user)
      result.STATUS = 1

      updateCount(user?.USER_ID, 0)
      return res.json(result)
    })
  })(req, res)
}

export const refreshToken = (req, res, next) => {
  const refreshToken = req.body.refreshToken

  if (refreshToken == undefined) {
    const errMsg = `can not refresh access token`
    logger.error(errMsg)
    return next(new Error(errMsg))
  }

  jwt.verify(refreshToken, secretKey, function (err, decoded) {
    if (err) {
      logger.error(`Token refresh error. [${err.message}]`)
      return res.status(400).send(err)
    }
    delete decoded["iat"]
    delete decoded["exp"]
    const result = _createToken(decoded)
    return res.json(result)
  })
}

export const logout = (req, res) => {
  const user = req.user
  res.status(200).send()
}

const _createToken = (user) => {
  const serverInfo = CL.get("serverInfo")
  const authInfo = CL.get("auth")
  const tokenExpireTime = (authInfo || {}).tokenExpireTime || "5m"
  const refreshTokenExpireTime = (authInfo || {}).refreshTokenExpireTime || "1h"

  const token = jwt.sign(user, secretKey, { expiresIn: tokenExpireTime })
  const refreshToken = jwt.sign(user, secretKey, {
    expiresIn: refreshTokenExpireTime
  })

  return { user, serverInfo, token, refreshToken }
}
