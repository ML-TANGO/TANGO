import React from "react";
import { useEffect, useState } from "react";

import '../../../CSS/project_management.css'

import graph_Image from "../../../images/bar-graph2x.png";

function VisualMain()
{
    const [server_ip, setServer_ip] = useState();

    /* 페이지 로드 완료시 호출 이벤트 */
    useEffect( () =>
    {
	    // setServer_ip('http://netron.app')
	    //
	    // vis2code port 8091
	    var host = window.location.hostname 
	    setServer_ip('http://' + host + ':8091')
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
                        <div className='title_left'>Visualization</div>
                    </div>
                </div>

                <div className='data_manage_content' style={{backgroundColor:'#DFDFDF', borderRadius:'5px'}}>
                     <iframe id='labelTool'  className='project_manage_content' title='labeling tool' src={server_ip} frameBorder='0' style={{ width:'100%', height:'100%',  borderRadius:'5px'}}></iframe>
                </div>

                {/*
                <div className='project_manage_content' style={{backgroundColor:'#ececec', borderRadius:'5px'}}>
                    <div style={{ padding:'20px 20px 20px 20px', overflow:'auto' }}>

                        <div style={{ padding:'50px 100px 50px 100px', width:'100%', height:'100%'}}>
                            <img src={graph_Image} alt="graph_Image" style={{ width:'100%', height:'100%'}}/>
                        </div>
                    </div>
                </div>
                 */}
            </div>
        </div>
        </>
    );
}

export default VisualMain;
