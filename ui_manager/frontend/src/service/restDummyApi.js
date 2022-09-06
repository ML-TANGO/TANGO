import axios from "axios";

import {getHeaderData, tokenExpiredCheck} from "./restApi_Base"

/* 서버 주소 */
//const dummy_server_ip_1 = 'http://192.168.0.179:8087'
//const dummy_server_ip_2 = 'http://192.168.0.179:8088'
//const dummy_server_ip_3 = 'http://192.168.0.179:8089'

//const address = "0.0.0.0"
//const address = "192.168.0.179"
//const address = window.location.hostname;

/* 신경망 생성 */
export function requestCreateNeural_Dummy(param)
{
    const header_info = getHeaderData()
    const server_ip = 'http://' + window.location.hostname + ':8087'

    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/create_neural/", param, {headers: header_info}).then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}

/* 이미지 생성 */
export function requestCreateImage_Dummy(param)
{
    const header_info = getHeaderData()
    const server_ip = 'http://' + window.location.hostname + ':8088'

    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/create_image/", param, {headers: header_info}).then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}

/* 이미지 배포 */
export function requestDeployImage_Dummy(param)
{
    const header_info = getHeaderData()
    const server_ip = 'http://' + window.location.hostname + ':8089'

    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/deploy_image/", param, {headers: header_info}).then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}