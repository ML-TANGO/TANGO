import axios from "axios";

import {server_ip, getHeaderData, getHeaderData_form, tokenExpiredCheck} from "./restApi_Base"

/* Container - Start */
export function requestContainerStart(container, uid, pid)
{
    let port = "";
    switch(container)
    {
        case 'yolo_e':
            port = '8090'
            break;
        case 'labelling':
            port = "8095"
            break;
        case 'bms':
            port = "8081";
            break;
        case 'viz':
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
        case 'deploy_cloud':
            port = "8088";
            break;
        case 'deploy_ondevice':
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
        case 'viz':
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
        case 'deploy_cloud':
            port = "8088";
            break;
        case 'deploy_ondevice':
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
