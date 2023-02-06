import axios from "axios";

import {server_ip, getHeaderData, getHeaderData_form, tokenExpiredCheck} from "./restApi_Base"

/* 데이터 셋 요청 */
//export function requestTargetCreate(param)
//{
//    const header_info = getHeaderData_form()
//    return new Promise( (resolve, reject) =>
//    {
//        axios.post( server_ip + "/api/target_create/", formData, {headers: header_info}).then((response) =>
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

/* 데이터 셋 리스트 요청 */
export function requestDatasetList(param)
{
    return new Promise( (resolve, reject) =>
    {
        var host = window.location.hostname;
        axios.post( 'http://' + host + ':8095' + "/api/dataset/getDataSetList", {headers: ""}).then((response) =>
        {
            resolve( response )
        })
        .catch(error =>
        {
            reject(error.response)
        });
    });
}
