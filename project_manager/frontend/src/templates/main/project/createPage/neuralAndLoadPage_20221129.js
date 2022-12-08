import React from "react";
import Cookies from "universal-cookie";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/combo_box.css'
import '../../../../CSS/stepProgress.css'
import '../../../../CSS/project_management.css'

import * as Request from "../../../../service/restProjectApi";
import * as RequestTarget from "../../../../service/restTargetApi";
import * as RequestContainer from "../../../../service/restContainerApi";
import * as RequestLabelling from "../../../../service/restLabellingApi";

import overall_up from "../../../../images/icons/icon_3x/chevron-up.png";
import overall_down from "../../../../images/icons/icon_3x/chevron-down.png";

import InformationForm from "./informationForm";
import ConfigForm from "./configForm";

import {Collapse} from 'react-collapse';
import Select from 'react-select';

function NeuralAndLoadPage({project_id, project_name, project_description})
{
    const inputMethodList = [
        { value: 'camera', label: 'Camera' },
        { value: 'mp4', label: 'MP4' },
        { value: 'picture', label: 'Picture' },
        { value: 'folder', label: 'Folder' }
    ]

    const outputMethodList = [
        { value: 'console', label: 'Console' },
        { value: 'graphic', label: 'Graphic' },
        { value: 'mp4', label: 'MP4' }
    ]

    const userEditList = [
        { value: 'yes', label: 'Yes' },
        { value: 'no', label: 'No' },
    ]

    const timerRef = useRef();

    const [autoCheck, setAutoCheck] = useState('');                    // 자동 신경망 생성 활성화 상태

    const [responseData, setResponseData] = useState('');              // 기존 선택 데이터

    const [container, setContainer] = useState('');                    // 신경망 생성 단계
    const [container_status, setContainer_status] = useState('');      // 신경망 생성 단계 상황

    const [currentWork, setCurrentWork] = useState('');                // 현재 작업 중인 컨테이너

    const [pannel, setPannel] = useState(false);                       // panel 창 상태
    const [config_pannel, setConfig_pannel] = useState(true);          // config_pannel 창 상태

    /* Information */
    const [dataset, setDataset] = useState('');                        // 데이터 셋 경로
    const [dataset_list, setDataset_list] = useState([]);              // 데이터 셋 리스트

    const [target, setTarget] = useState('');                          // 타겟 변경
    const [target_list, setTarget_list] = useState([]);                // 타겟 리스트

    /* Configuration - Task Type*/
    const [taskType, setTaskType] = useState('');                      // 데이터 셋 경로

    /* Configuration - AutoNN */
    const [datasetFile, setDatasetFile] = useState('dataset.yaml');    // dataset file yaml 파일
    const [baseModel, setBaseModel] = useState('basemodel.yaml');      // Base Mode yaml 파일

    /* Configuration - Nas Type */
    const [nasType, setNasType] = useState('');                        // 데이터 셋 경로

    /* Configuration - Deploy Configuration */
    const [weightLevel, setWeightLevel] = useState(0);                 // weightLevel
    const [precisionLevel, setPrecisionLevel] = useState(0);           // precisionLevel
    const [processingLib, setProcessingLib] = useState('cv2');         // processingLib
    const [userEdit, setUserEdit] = useState('');                      // userEdit
    const [inputMethod, setInputMethod] = useState('');                // inputMethod
    const [inputDataPath, setInputDataPath] = useState('/data');       // inputDataPath
    const [outputMethod, setOutputMethod] = useState('');              // outputMethod

    useEffect( () =>
    {
        // 프로젝트 정보 수신
        Request.requestProjectInfo(project_id).then(result =>
        {
            projectContentUpdate(result.data);

            // 타이머 기능
//            if(result.data['container_status'] === 'running')
//            {
//                startTimer()
//            }

            // project_info.yaml 파일을 생성 하지 않은 경우
            if(result.data['container'] === '')
            {
                setPannel(true)
            }
        })
        .catch(error =>
        {
            console.log('project info get error')
        });

        // unmount
        return() => {
            //console.log('unmount')
            clearInterval(timerRef.current)
        }
    }, []);

    // 프로젝트 정보 업데이트
    const projectContentUpdate = (data) =>
    {
        // DB에 저장된 데이터
        setResponseData(data)

        if(data['target'] !== '') setTarget(parseInt(data['target']))  // 선택 타겟 정보
        if(data['dataset'] !== ''){
            setDataset(data['dataset'])         // 데이터 셋 정보
            setTaskType(data['task_type'])
        }

        if(data['nas_type'] !== '') setNasType(data['nas_type'])

        if(data['deploy_weight_level'] !== '') setWeightLevel(parseInt(data['deploy_weight_level']))
        if(data['deploy_precision_level'] !== '') setPrecisionLevel(parseInt(data['deploy_precision_level']))

        const im_index = inputMethodList.findIndex(im => im.value === data['deploy_input_method'])
        if(im_index !== -1) setInputMethod(inputMethodList[im_index])

        const om_index = outputMethodList.findIndex(om => om.value === data['deploy_output_method'])
        if(om_index !== -1) setOutputMethod(outputMethodList[om_index])

        const ue_index = userEditList.findIndex(ue => ue.value === data['deploy_user_edit'])
        if(ue_index !== -1) setUserEdit(userEditList[ue_index])

        setContainer(data['container'])                 // 진행중 컨테이너
        setContainer_status(data['container_status'])   // 진행중 컨테이너 상태
    }

    const test = () => {
        console.log('test')
    }

    const startTimer = () =>
    {
        if(!timerRef.current)
        {
            // 타이머 시작 - 10초 주기
            timerRef.current = setInterval(test, 10000)
        }
    }

    const stopTimer = () =>
    {
        if(timerRef.current)
        {
            clearInterval(timerRef.current)
            timerRef.current = null;
        }
    }

    // 컨테이너 상태 수신
    const get_container_status = () =>
    {
        const param = {
            'project_id': project_id,
        };
        Request.requestContainerStatus(param).then(result =>
        {
            const data = result.data;

            setContainer(data.container)
            setContainer_status(data.container_status)

            if(data.container_status !== '')
            {
                // TODO : 타이머 중지
            }
        })
        .catch(error =>
        {
            console.log('container status get error')
        });
    }

    // 타겟 정보 확인
    const get_target_info = (id) =>
    {
        const findIndex = target_list.findIndex(v => v.id === id)

        if (findIndex !== -1)
        {
            return target_list[findIndex].info
        }
    }

    // Information 창 보이기 숨기기
    const accordionButtonClick = () =>
    {
        if(pannel === true)
        {
            setPannel(false)
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_down + "')";
        }
        else{
            setPannel(true)
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_up + "')";
        }
    }

    // Configuration 창 보이기 숨기기
    const configAccordionButtonClick = () =>
    {
        if(config_pannel === true)
        {
            setConfig_pannel(false)
            document.getElementById('config_overall_icon').style.backgroundImage = "url('" + overall_down + "')";
        }
        else{
            setConfig_pannel(true)
            document.getElementById('config_overall_icon').style.backgroundImage = "url('" + overall_up + "')";
        }
    }


    // 'Run' 버튼 클릭 이벤트
    const runButtonClick = () =>
    {
        const param = {
            'project_id' : project_id,
            'project_target' : target,
            'project_dataset' : dataset,
            'task_type': taskType,
            'autonn_dataset_file': datasetFile,
            'autonn_base_model': baseModel,
            'nas_type': nasType,
            'deploy_weight_level': weightLevel,
            'deploy_precision_level': precisionLevel,
            'deploy_processing_lib': processingLib,
            'deploy_user_edit': userEdit !== '' ? userEdit.value : '',
            'deploy_input_method': inputMethod !== '' ? inputMethod.value : '' ,
            'deploy_input_data_path': inputDataPath,
            'deploy_output_method': outputMethod !== '' ? outputMethod.value : '',
        };

        if(container === '' || container === 'init')
        {
            neuralCreate(param)
        }
        else
        {
            var result = window.confirm("새로운 신경망을 생성 하시겠습니까?")
            if(result === true) neuralCreate(param)
        }
    };

    // 신경망 생성 시작
    const neuralCreate = (param) =>
    {
//        console.log(param)

        // 데이터베이스 업데이트
        Request.requestProjectUpdate(param).then(result =>
        {
            alert('신경망 생성 준비 완료');

            // 프로젝트 정보 수신
            Request.requestProjectInfo(project_id).then(res =>
            {
                projectContentUpdate(res.data);
            })
            .catch(error =>
            {
                console.log('project info get error')
            });
        })
        .catch(error =>
        {
            console.log(error);
        });
    }

    // Base Model Select: 8081 port
    const bmsButtonClick = () =>
    {
        // console.log("bmsButtonClick")

        containerStart('bms');

        setCurrentWork('Base Model Select');

        status_result_update('Base Model Select - Start 요청')
    }

    // Visualization: 8091 port
    const visualButtonClick = () =>
    {
        //console.log("visualButtonClick")

        containerStart('viz');

        setCurrentWork('Visualization');

        status_result_update('Visualization - Start 요청')
    }

    // Auto NN : Backbone Nas = 8087 port / Neck Nas = 8089 port
    const autoNNButtonClick = () =>
    {
        //console.log("autoNNButtonClick")

        console.log(responseData)

        if(responseData.nas_type === 'neck_nas')
        {
            containerStart('autonn_nk');

            setCurrentWork('Auto NN : Neck NAS');

            status_result_update('Auto NN : Neck NAS - Start 요청')
        }
        else
        {
            containerStart('autonn_bb');

            setCurrentWork('Auto NN : Backbone NAS');

            status_result_update('Auto NN : Backbone NAS - Start 요청')
        }
    }

    // Image Gen : 8888 port
    const imageGenButtonClick = () =>
    {
        //console.log("imageGenButtonClick")

        containerStart('code_gen');

        setCurrentWork('Image Generate');

        status_result_update('Image Generate - Start 요청')
    }

    // Image Deploy : target_info 'PC or Cloud ' 8088 port / onDevice 8891 port
    const imageDeployButtonClick = () =>
    {
        //console.log("imageDeployButtonClick")

        const target_id = responseData.target;
        const indexInfo = target_list.findIndex(v => v.id === parseInt(target_id))

        if(indexInfo === -1)
        {
            alert("타겟 정보가 존재하지 않습니다.")
            return;
        }

        const targetData = target_list[indexInfo]

        if(targetData.info === 'ondevice')
        {
            containerStart('deploy_ondevice');

            setCurrentWork('Image Deploy : On Device');

            status_result_update('Image Deploy : On Device - Start 요청')
        }
        else
        {
            containerStart('deploy_cloud');

            setCurrentWork('Image Deploy : ' + targetData.info);

            status_result_update('Image Deploy : ' + targetData.info + ' - Start 요청')
        }
    }

    // Run Image 버튼 클릭
    const runImageButtonClick = () =>
    {
        setCurrentWork('Run Image');

        status_result_update('Run Image : Start 요청')
    }

    // 컨테이너 실행
    const containerStart = (name) =>
    {
        // project_info.yaml 파일 생성을 하지않은 경우
        if(container === '')
        {
            alert('프로젝트 정보를 생성해주세요')
            return;
        }

        const cookies = new Cookies();
        var user = cookies.get('userinfo')

        RequestContainer.requestContainerStart(name, user, project_id).then(result =>
        {
            status_result_update(JSON.stringify(result.data))
        })
        .catch(error =>
        {
            status_result_update(error)
        });
    }

    // 상태창 업데이트
    const status_result_update = (text) =>
    {
        const textArea = document.getElementById('status_result');
        textArea.value += (text + '\n');
        textArea.scrollTop = textArea.scrollHeight;
    }

    return (
        <>

        {/* 프로젝트 생성 - 신경망 생성 폼 */}
        <div className='project_manage_container'>
            <div className='project_manage_content' style={{width:'100%'}}>
                <div id='project_top' className='project_top'  style={{padding:'0px 0px 0px 0px', height:'100%'}}>

                    <div id="accordion" className="accordion" onClick={ ()=> accordionButtonClick() } style={{height:'40px', position:'static', backgroundColor:'#303030', borderRadius:pannel === false? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>
                        <span style={{fontSize:'16px', color:'white'}}>Information </span>
                        <div id="overall_icon" className="overall_icon" style={{backgroundImage:pannel === false ? "url('" + overall_down + "')" : "url('" + overall_up + "')"}}></div>
                    </div>

                    <Collapse isOpened={pannel}>
                        <InformationForm
                            dataset={dataset} setDataset={setDataset}
                            dataset_list={dataset_list} setDataset_list={setDataset_list}
                            target={target} setTarget={setTarget}
                            target_list={target_list} setTarget_list={setTarget_list}
                            setTaskType={setTaskType}/>
                    </Collapse>

                    <div id="accordion" className="accordion" onClick={ ()=> configAccordionButtonClick() } style={{height:'40px', marginTop:'20px', position:'static', backgroundColor:'#303030', borderRadius:config_pannel === false? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>
                        <span style={{fontSize:'16px', color:'white'}}>Configuration </span>
                        <div id="config_overall_icon" className="config_overall_icon" style={{backgroundImage:config_pannel === false ? "url('" + overall_down + "')" : "url('" + overall_up + "')"}}></div>
                    </div>

                    <Collapse isOpened={config_pannel}>
                        <ConfigForm
                            inputMethodList={inputMethodList} outputMethodList={outputMethodList} userEditList={userEditList}
                            target={target}
                            taskType={taskType} setTaskType={setTaskType}
                            datasetFile={datasetFile} setDatasetFile={setDatasetFile}
                            baseModel={baseModel} setBaseModel={setBaseModel}
                            nasType={nasType} setNasType={setNasType}
                            weightLevel={weightLevel} setWeightLevel={setWeightLevel}
                            precisionLevel={precisionLevel} setPrecisionLevel={setPrecisionLevel}
                            processingLib={processingLib} setProcessingLib={setProcessingLib}
                            userEdit={userEdit} setUserEdit={setUserEdit}
                            inputMethod={inputMethod} setInputMethod={setInputMethod}
                            inputDataPath={inputDataPath} setInputDataPath={setInputDataPath}
                            outputMethod={outputMethod} setOutputMethod={setOutputMethod}
                            get_target_info={get_target_info}/>
                    </Collapse>

                    <div id='runButtonArea' className='button' style={{marginTop:'20px', textAlign:'center'}}>

                        <span>
                            <span>Auto Create</span>
                            <input type="checkbox"
                                onChange={(value) => setAutoCheck(value)}
                                style={{width:'20px', marginRight:'30px'}}/>
                        </span>


                        {get_target_info(target) === 'pc' || get_target_info(target) === 'cloud' ?
                            <>
                            { target !== '' && dataset !== '' && taskType !== '' && nasType !== '' && userEdit !== '' && inputMethod !== '' && outputMethod !== ''?
                                <button onClick={ ()=> runButtonClick() } style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor: '#4A80FF', color:'white'}}>신경망 자동 생성</button>
                            :
                                <button style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#707070', color:'white'}} readOnly>신경망 자동 생성</button>
                            }
                            </>
                        :
                            <>
                            { target !== '' && dataset !== '' && taskType !== '' && nasType !== ''?
                                <button onClick={ ()=> runButtonClick() } style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#4A80FF', color:'white'}}>신경망 자동 생성</button>
                            :
                                <button style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#707070', color:'white'}} readOnly>신경망 자동 생성</button>
                            }
                            </>
                        }
                    </div>
                </div>

                <div id='project_bottom' className='project_bottom'  style={{padding:'20px 0px 0px 0px', height:'100%', marginBottom:'0px'}}>
                    <div className='create_neural_network' style={{ backgroundColor:'#303030', borderRadius:'5px', height:'100%', padding:'10px 20px 20px 20px'}}>
                        <div style={{marginBottom:'10px', display:'flex'}}>
                            <span style={{fontSize:'16px', color:'white'}}>Current Work - </span>
                            <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}>[ </span>
                            <span style={{color:'#4A80FF'}}>{currentWork}</span>
                            <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}> ]</span>
                        </div>

                        <div className='status_level' style={{backgroundColor:'white', padding:'20px 0px', borderRadius:'5px', display:'grid', gridTemplateRows:'1fr 1fr'}}>
                            <div className="stepper-wrapper2" id='progressbar' style={{gridRow:'1/2'}}>
                                <div className="stepper-item2 non-select" id='progress_1'>
                                    <button className="step-counter2"
                                            style={{backgroundColor:container === '' ? 'gray' : '#4A80FF', color:'white'}}
                                            onClick={() => bmsButtonClick()}>BMS</button>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_2'>
                                    <button className="step-counter2"
                                            style={{backgroundColor:container === '' ? 'gray' : '#4A80FF', color:'white'}}
                                            onClick={() => autoNNButtonClick()}>AutoNN</button>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_3'>
                                    <button className="step-counter2"
                                            style={{backgroundColor:container === '' ? 'gray' : '#4A80FF', color:'white'}}
                                            onClick={() => imageGenButtonClick()}>Image Gen</button>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_4'>
                                    <button className="step-counter2"
                                            style={{backgroundColor:container === '' ? 'gray' : '#4A80FF', color:'white'}}
                                            onClick={() => imageDeployButtonClick()}>Image Deploy</button>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_5'>
                                    <button className="step-counter2"
                                            style={{backgroundColor:container === '' ? 'gray' : '#4A80FF', color:'white'}}
                                            onClick={() => runImageButtonClick()}>Run Image</button>
                                </div>
                            </div>

                            <div className="stepper-wrapper3" id='progressbar' style={{marginTop:'10px', gridRow:'2/3', display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr 1fr'}}>
                                <div className="stepper-item3 non-select" id='progress_0' style={{gridColumn:'1/3'}}>
                                    <button className="step-counter2"
                                            style={{backgroundColor:container === '' ? 'gray' : '#4A80FF', color:'white'}}
                                            onClick={() => visualButtonClick()}>Visualization</button>
                                </div>
                            </div>
                        </div>

                        <div className='status_log' style={{color:'white', height:'auto', overflow:'auto', minHeight:'200px', padding:'20px 0px 0px 0px'}}>
                            <div style={{ border:'2px solid white', borderRadius:'5px', backgroundColor:'white', position:'relative', width:'100%', height:'100%'}}>
                                <textarea id='status_result' style={{backgroundColor:'white', color:'black', resize:'none', width:'100%', height:'100%',borderRadius:'5px', border:'0px'}} readOnly/>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        </>
    );
}

export default NeuralAndLoadPage;
