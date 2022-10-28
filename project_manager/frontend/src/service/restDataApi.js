import axios from "axios";
import Cookies from "universal-cookie";

import {server_ip, getHeaderData, tokenExpiredCheck} from "./restApi_Base"


/* 프로젝트 리스트 요청 */
export function requestGetServerIp()
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.get( server_ip + "/api/get_server_ip/",  {headers: header_info}).then((response) =>
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