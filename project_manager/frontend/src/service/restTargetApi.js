import axios from "axios";

import {server_ip, getHeaderData, getHeaderData_form, tokenExpiredCheck} from "./restApi_Base"

/* 타겟 생성 요청 */
export function requestTargetCreate(param)
{
    const formData = new FormData();
    formData.append("name", param.name);
//    formData.append("image", new Blob([param.image], { type: param.image.type } ) );
    formData.append("image", param.image);
    formData.append("cpu", param.cpu);
    formData.append("gpu", param.gpu);
    formData.append("memory", param.memory);
    formData.append("model", param.model);

    const header_info = getHeaderData_form()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/target_create/", formData, {headers: header_info}).then((response) =>
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

/* 타겟 리스트 요청 */
export function requestTargetList()
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.get( server_ip + "/api/target_read/",  {headers: header_info}).then((response) =>
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

/* 타겟 수정 요청 */
export function requestTargetUpdate(param)
{
    const formData = new FormData();
    formData.append("id", param.id);
    formData.append("name", param.name);
//    formData.append("image", new Blob([param.image], { type: param.image.type } ) );
    formData.append("image", param.image);
    formData.append("cpu", param.cpu);
    formData.append("gpu", param.gpu);
    formData.append("memory", param.memory);
    formData.append("model", param.model);

    const header_info = getHeaderData_form()
    return new Promise( (resolve, reject) =>
    {
        axios.put( server_ip + "/api/target_update/",  formData, {headers: header_info}).then((response) =>
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

/* 타겟 삭제 요청 */
export function requestTargetDelete(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.delete( server_ip + "/api/target_delete/", {headers: header_info, data:{id: param}}).then((response) =>
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

