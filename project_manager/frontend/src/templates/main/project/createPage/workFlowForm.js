import React from "react";
import Cookies from "universal-cookie";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/project_management.css'
import '../../../../CSS/stepProgress_v2.css'
import '../../../../CSS/loader.css'

import * as RequestContainer from "../../../../service/restContainerApi";

function WorkFlowForm({project_id,
                       yamlMake, yamlData,
                       container, container_status,
                       containerStatusUpdate, step_progress_bar_update})
{
//    useEffect( () =>
//    {
//        console.log(container)
//    }, []);

    const getCurrentWork = () =>
    {
        let current_work = ''

        switch(container)
        {
            case 'bms' :
                current_work = 'Base Model Select'
                break;
            case 'vis2code' :
                current_work = 'Visualization'
                break;
            case 'autonn_nk' :
                current_work = 'Auto NN'
                break;
            case 'autonn_bb' :
                current_work = 'Auto NN'
                break;
            case 'code_gen' :
                current_work = 'Image Gen'
                break;
            case 'cloud_deployment' :
                current_work = 'Image Deploy'
                break;
            case 'ondevice_deployment' :
                current_work = 'Image Deploy'
                break;
            case 'run_image' :
                current_work = 'Run Image'
                break;
            default:
                break;
        }
        return current_work
    }

    const container_confirm = (con, word) =>
    {
        var result = window.confirm(word)
        if(result === true)
        {
            containerStart(con);
        }

        return result
    }

    // Base Model Select : 8081 port
    const bmsButtonClick = () =>
    {
        const bms_progress = document.getElementById('progress_1');

        if(bms_progress.className.includes('before'))
        {
            return
        }
        else if(bms_progress.className.includes('ready'))
        {
            containerStart('bms');
            status_result_update('Base Model Select - Start 요청')
        }
        else
        {
            const result_check = container_confirm('bms', 'Base Model Select를 다시 실행 하시겠습니까?')
            if(result_check) status_result_update('Base Model Select - Start 요청')
        }
    }

    // Visualization: 8091 port
    const visualButtonClick = () =>
    {
        const vis2code_progress = document.getElementById('progress_2');

        if(vis2code_progress.className.includes('before'))
        {
            return
        }
        else if(vis2code_progress.className.includes('ready'))
        {
            containerStart('vis2code');
            status_result_update('Visualization - Start 요청')
        }
        else
        {
            const result_check = container_confirm('vis2code', 'Visualization를 다시 실행 하시겠습니까?')
            if(result_check) status_result_update('Visualization - Start 요청')
        }
    }

    // Auto NN : Backbone Nas = 8087 port / Neck Nas = 8089 port
    const autoNNButtonClick = () =>
    {
        const autonn_progress = document.getElementById('progress_3');

        if(autonn_progress.className.includes('before'))
        {
            return
        }
        else if(autonn_progress.className.includes('ready'))
        {
            if(yamlData.nas_type === 'neck_nas')
            {
                containerStart('autonn_nk');
                status_result_update('Auto NN : Neck NAS - Start 요청')
            }
            else
            {
                containerStart('autonn_bb');
                status_result_update('Auto NN : Backbone NAS - Start 요청')
            }
        }
        else
        {
            if(yamlData.nas_type === 'neck_nas')
            {
                const result_check = container_confirm('autonn_nk', 'Auto NN을 다시 실행 하시겠습니까?')
                if(result_check) status_result_update('Auto NN : Neck NAS - Start 요청')
            }
            else
            {
                const result_check = container_confirm('autonn_bb', 'Auto NN을 다시 실행 하시겠습니까?')
                if(result_check) status_result_update('Auto NN : Backbone NAS - Start 요청')
            }
        }
    }

    // Image Gen : 8888 port
    const imageGenButtonClick = () =>
    {
        const codegen_progress = document.getElementById('progress_4');

        if(codegen_progress.className.includes('before'))
        {
            return
        }
        else if(codegen_progress.className.includes('ready'))
        {
            containerStart('code_gen');
            status_result_update('Image Generate - Start 요청')
        }
        else
        {
            const result_check = container_confirm('code_gen', 'Image Generate를 다시 실행 하시겠습니까?')
            if(result_check) status_result_update('Image Generate - Start 요청')
        }
    }

    // Image Deploy : target_info 'PC or Cloud ' 8088 port / onDevice 8891 port
    const imageDeployButtonClick = () =>
    {
        const deploy_progress = document.getElementById('progress_5');

        if(deploy_progress.className.includes('before'))
        {
            return
        }
        else if(deploy_progress.className.includes('ready'))
        {
            if(yamlData.target.info === 'ondevice')
            {
                containerStart('ondevice_deployment');
            }
            else
            {
                containerStart('cloud_deployment');
            }
            status_result_update('Image Deploy : ' + yamlData.target.info + ' - Start 요청')
        }
        else
        {
            if(yamlData.target.info === 'ondevice')
            {
                const result_check = container_confirm('ondevice_deployment', 'Image Deploy 다시 실행 하시겠습니까?')
                if(result_check) status_result_update('Image Deploy : ' + yamlData.target.info + ' - Start 요청')
            }
            else
            {
                const result_check = container_confirm('cloud_deployment', 'Image Deploy 다시 실행 하시겠습니까?')
                if(result_check) status_result_update('Image Deploy : ' + yamlData.target.info + ' - Start 요청')
            }
        }
    }

    // Run Image 버튼 클릭
    const runImageButtonClick = () =>
    {
        const run_image_progress = document.getElementById('progress_6');

        if(run_image_progress.className.includes('before'))
        {
            return
        }
        else if(run_image_progress.className.includes('ready'))
        {
            alert('현재 서비스 준비중입니다.')
        }
    }

    // 컨테이너 실행
    const containerStart = (name) =>
    {
        const cookies = new Cookies();
        var user = cookies.get('userinfo')

        RequestContainer.requestContainerStart(name, user, project_id).then(result =>
        {
            containerStatusUpdate(name, 'run')
            step_progress_bar_update(name, 'run')

            status_result_update(JSON.stringify(result.data))
        })
        .catch(error =>
        {
            status_result_update(error)

            containerStatusUpdate(name, 'fail')
            step_progress_bar_update(name, 'fail')
        });
    }

    // 상태창 업데이트
    const status_result_update = (text) =>
    {
        const textArea = document.getElementById('status_result');
        textArea.value += (text + '\n');
        textArea.scrollTop = textArea.scrollHeight;
    }

    // 컨테이너 상태 요청
    const statusRequestClick = () =>
    {
        status_result_update(container + ' : 상태 요청')

        const cookies = new Cookies();
        var user = cookies.get('userinfo')

        RequestContainer.requestContainerStatusCheck(container, user, project_id).then(result =>
        {
            status_result_update(JSON.stringify(result.data))
        })
        .catch(error =>
        {
            status_result_update(error)
        });
    }


    return (
        <>
        <div id='project_bottom' className='project_bottom'  style={{padding:'20px 0px 0px 0px', height:'100%', marginBottom:'0px'}}>
            <div className='create_neural_network' style={{ backgroundColor:'#303030', borderRadius:'5px', height:'100%', padding:'10px 20px 20px 20px'}}>
                <div style={{marginBottom:'10px', display:'flex'}}>
                    <span style={{fontSize:'16px', color:'white'}}>Current Work - </span>
                    <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}>[ </span>
                    <span style={{color:'#4A80FF'}}>{getCurrentWork()}</span>
                    <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}> ]</span>

                    {yamlMake === true &&
                        <button className="" onClick={() => statusRequestClick()} style={{height:'30px', width:'150px', fontSize:'14px', backgroundColor:'white', borderRadius:'5px', border:'0px', color:'black'}}>Status Request</button>
                    }
                </div>

                        <div className='status_level' style={{backgroundColor:'white', padding:'20px 0px', borderRadius:'5px', display:'grid', gridTemplateRows:'1fr 1fr'}}>
                            <div className="stepper-wrapper" id='progressbar_main' style={{gridRow:'1/2'}}>
                                <div className="stepper-item before" id='progress_1'>
                                    <button className="step-counter" onClick={() => bmsButtonClick()}>BMS
                                    {container.includes('bms') && container_status.includes('run') &&
                                        <span className="loader"></span>
                                    }
                                    </button>
                                </div>
                                <div className="stepper-item before" id='progress_3'>
                                    <button className="step-counter" onClick={() => autoNNButtonClick()}>AutoNN
                                    {container.includes('autonn') && container_status.includes('run') &&
                                        <span className="loader"></span>
                                    }
                                    </button>
                                </div>
                                <div className="stepper-item before" id='progress_4'>
                                    <button className="step-counter" onClick={() => imageGenButtonClick()}>Image Gen
                                    {container.includes('code_gen') && container_status.includes('run') &&
                                        <span className="loader"></span>
                                    }
                                    </button>
                                </div>
                                <div className="stepper-item before" id='progress_5'>
                                    <button className="step-counter" onClick={() => imageDeployButtonClick()}>Image Deploy
                                    {container.includes('deployment') && container_status.includes('run') &&
                                        <span className="loader"></span>
                                    }
                                    </button>
                                </div>
                                <div className="stepper-item before" id='progress_6'>
                                    <button className="step-counter" onClick={() => runImageButtonClick()}>Run Image</button>
                                </div>
                            </div>

                            <div className="stepper-wrapper2" id='progressbar_sub' style={{marginTop:'10px', gridRow:'2/3', display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr 1fr'}}>
                                <div className="stepper-item before" id='progress_2' style={{gridColumn:'1/3'}}>
                                    <button className="step-counter" onClick={() => visualButtonClick()}>Visualization
                                    {container.includes('vis2code') && container_status.includes('run') &&
                                        <span className="loader"></span>
                                    }
                                    </button>
                                </div>
                            </div>
                        </div>

                <div className='status_log' style={{color:'white', height:'auto', minHeight:'200px', padding:'20px 0px 0px 0px'}}>
                    <div style={{ border:'2px solid white', borderRadius:'5px', backgroundColor:'white', position:'relative', width:'100%', height:'100%'}}>
                        <textarea id='status_result' style={{backgroundColor:'white', color:'black', resize:'none', width:'100%', height:'100%',borderRadius:'5px', border:'0px'}} readOnly/>
                    </div>
                </div>
            </div>
        </div>
        </>
    );
}

export default WorkFlowForm;