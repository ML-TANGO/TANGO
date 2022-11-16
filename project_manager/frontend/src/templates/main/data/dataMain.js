import React from "react";
import { useEffect, useState } from "react";

import '../../../CSS/project_management.css'

import * as Request from "../../../service/restDataApi";

function DataMain()
{
    const [server_ip, setServer_ip] = useState();

    /* 페이지 로드 완료시 호출 이벤트 */
    useEffect( () =>
    {
        var host = window.location.hostname
        setServer_ip('http://' + host + ':8086')

        // 서버 ip 주소 요청
//        Request.requestGetServerIp().then(result =>
//        {
//            var host = window.location.hostname
//            setServer_ip('http://' + host + ':' + result.data['port'])
//        })
//        .catch(error =>
//        {
//            alert('get ip info error')
//        });

    }, []);


    return (
        <>
        <div className='manage_list_container_sub'>
            {/* 데이터 셋 메인 페이지 */}
            <div className='manage_container'>

                {/* 데이터 셋 메인 페이지 - 헤더 */}
                <div className='manage_header' style={{width:'100%'}}>
                    {/* 이동경로 레이아웃 */}
                    <div className='path'></div>

                    <div className='title'>
                        <div className='title_left'>Data Management</div>
                    </div>
                </div>

                <div className='data_manage_content' style={{backgroundColor:'#DFDFDF', borderRadius:'5px'}}>
                     <iframe id='labelTool'  className='project_manage_content' title='labeling tool' src={server_ip} frameBorder='0' style={{ width:'100%', height:'100%', borderRadius:'5px'}}></iframe>
                </div>

            </div>
        </div>
        </>
    );
}

export default DataMain;