import axios from "axios";
import Cookies from "universal-cookie";

import {server_ip, getHeaderData, tokenExpiredCheck} from "./restApi_Base"


/* 프로젝트 리스트 요청 */
export function requestProjectList()
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.get( server_ip + "/api/project_list_get/",  {headers: header_info}).then((response) =>
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

/* 프로젝트 생성 */
export function requestProjectCreate(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/project_create/", {'project_name': param['name'], 'project_description': param['description']}, { headers:header_info } ).then((response) =>
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

/* 프로젝트 이름 수정 */
export function requestProjectRename(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/project_rename/", {'id': param['id'], 'name':param['name']}, { headers:header_info } ).then((response) =>
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

/* 프로젝트 설명 수정 */
export function requestProjectDescriptionModify(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/project_description_update/", {'id': param['id'], 'description':param['description']}, { headers:header_info } ).then((response) =>
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

/* 프로젝트 삭제 */
export function requestProjectDelete(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/project_delete/", { id:param }, { headers:header_info } ).then((response) =>
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

/* 프로젝트 정보 조회 */
export function requestProjectInfo(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/project_info/", { id:param }, { headers:header_info } ).then((response) =>
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

/* 프로젝트 업데이트 */
export function requestProjectUpdate(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/project_update/", param, { headers:header_info } ).then((response) =>
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

/* 프로젝트 타겟 yaml 파일 생성 */
export function requestCreateTargetYaml(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/target_check/",  param, {headers: header_info}).then((response) =>
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


/* 데이터셋 경로 유효성 검사 */
export function requestDataSetAvailabilityCheck(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/dataset_check/", {name : param}, {headers: header_info}).then((response) =>
        {
            resolve( response.data )
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


/* 컨테이너 상태 요청 */
export function requestContainerStatus(param)
{
    const header_info = getHeaderData()
    return new Promise( (resolve, reject) =>
    {
        axios.post( server_ip + "/api/status_result/",  param, {headers: header_info}).then((response) =>
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
