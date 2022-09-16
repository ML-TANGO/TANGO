import axios from "axios";
import Cookies from "universal-cookie";

import {server_ip, getHeaderData, tokenExpiredCheck} from "./restApi_Base"

/* 로그인 요청 - OAuth Token 발행 */
export function requestLogin(id, pw)
{
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/login/",  {'user_id': id, 'password': pw} ).then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            const result = tokenExpiredCheck(error.response)

            if(result === false)
            {
                reject(error.response)
            }
        });
    });
}

/* 로그아웃 요청 - OAuth Token 취소 */
//export function requestLogout()
//{
//    const header_data = getHeaderData()
//    return new Promise( (resolve, reject) =>
//    {
//        axios.post( server_ip + "/api/logout/", {}, { headers:header_data }).then((response) =>
//        {
//            resolve( response )
//        })
//        .catch(error =>
//        {
//            reject(error.response)
//        });
//    });
//}

/* 아이디 중복 확인 */
export function requestSignUpDuplicate(param)
{
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/user_id_check", {'id': param}).then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}

/* 회원가입 요청 */
export function requestSignUp(param)
{
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/signup", {'id': param['id'], 'email': param['email'], 'password':param['password']}).then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}