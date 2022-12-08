import axios from "axios";

import {server_ip, getHeaderData, getHeaderData_form, tokenExpiredCheck} from "./restApi_Base"

/* Container - Start */
export function requestContainerStart(container, uid, pid)
{
    let port = "";
    switch(container)
    {
        case 'bms':
            port = "8081";
            break;
        case 'vis2code':
            port = "8091";
            break;
        case 'autonn_bb':
            port = "8087";
            break;
        case 'autonn_nk':
            port = "8089";
            break;
        case 'code_gen':
            port = "8888";
            break;
        case 'cloud_deployment':
            port = "8088";
            break;
        case 'ondevice_deployment':
            port = "8891";
            break;
        default:
            break;
    }

    return new Promise( (resolve, reject) =>
    {
        var host = window.location.hostname;
        axios.get( 'http://' + host + ':' + port + "/start", {
            params: {
                user_id: uid,
                project_id: pid
            }},
            { withCredentials: true })
        .then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}

/* Container - status request */
export function requestContainerStatusCheck(container, uid, pid)
{
    let port = "";
    switch(container)
    {
        case 'bms':
            port = "8081";
            break;
        case 'vis2code':
            port = "8091";
            break;
        case 'autonn_bb':
            port = "8087";
            break;
        case 'autonn_nk':
            port = "8089";
            break;
        case 'code_gen':
            port = "8888";
            break;
        case 'cloud_deployment':
            port = "8088";
            break;
        case 'ondevice_deployment':
            port = "8891";
            break;
        default:
            break;
    }

    return new Promise( (resolve, reject) =>
    {
        var host = window.location.hostname;
        axios.get( 'http://' + host + ':' + port + "/status_request", {
            params: {
                user_id: uid,
                project_id: pid
            }},
            { withCredentials: true })
        .then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}

/* 컨테이너 상태 요청 */
//export function requestContainerStatus(param)
//{
//    const header_info = getHeaderData()
//    return new Promise( (resolve, reject) =>
//    {
//        axios.post( server_ip + "/api/status_result/",  param, {headers: header_info}).then((response) =>
//        {
//            resolve( response )
//        })
//        .catch(error =>
//        {
//            const result = tokenExpiredCheck(error.response)
//
//            if(result === false)
//            {
//                reject(error.response)
//            }
//        });
//    });
//}

/* 컨테이너 상태 업데이트 */
export function requestContainerStatusUpdate(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/status_update/",  param, {headers: header_info}).then((response) =>
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
