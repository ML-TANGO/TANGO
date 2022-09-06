//import axios from "axios";
import Cookies from "universal-cookie";

/* 서버 주소 */
export const server_ip = ''
//export const server_ip = 'http://127.0.0.1:8000'
//export const server_ip = 'http://192.168.0.179:8001'
//export const server_ip = 'http://192.168.0.179:8002'


export function getHeaderData()
{
    /* 웹 브라우저 쿠키 정보 가져오기 */
    const token = new Cookies().get('DF_TOKEN')

//    const token = sessionStorage.getItem("DF_TOKEN");

    /* 서버에 전달할 header 정보 */
    const header_info = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
    }

    return header_info
}

/* 토큰 만료 시 쿠키 삭제 후 페이지 이동 */
export function tokenExpiredCheck(error)
{
    if(error['status'] === 401)
    {
        console.log(error)

        alert('토큰 만료')

        const cookie_info = new Cookies()
        cookie_info.remove('userinfo', {path:'/'});
        cookie_info.remove('DF_TOKEN', {path:'/'});

        window.location.replace('/')

        return true
    }

    return false
}




