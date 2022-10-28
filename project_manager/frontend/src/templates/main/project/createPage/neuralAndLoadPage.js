import React from "react";
import Cookies from "universal-cookie";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/combo_box.css'
import '../../../../CSS/stepProgress.css'
import '../../../../CSS/project_management.css'

import * as Request from "../../../../service/restProjectApi";
//import * as RequestDummy from "../../../../service/restDummyApi";
//import * as RequestData from "../../../../service/restDataApi";
import * as RequestTarget from "../../../../service/restTargetApi";
import * as RequestContainer from "../../../../service/restContainerApi";

import overall_up from "../../../../images/icons/icon_3x/chevron-up.png";
import overall_down from "../../../../images/icons/icon_3x/chevron-down.png";

import data_th_1 from "../../../../images/thumbnail/data_th_1.PNG";         // 칫솔
import data_th_2 from "../../../../images/thumbnail/data_th_2.PNG";         // 용접 파이프
import data_th_3 from "../../../../images/thumbnail/data_th_3.PNG";         // 실생활
import data_th_4 from "../../../../images/thumbnail/data_th_4.PNG";         // 폐결핵 판독

import * as TargetPopup from "../../../../components/popup/targetPopup";

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

    const [oriData, setOriData] = useState('');                        // originalData

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

    /* 타겟 생성 정보 */
    const [targetName, setTargetName] = useState('');                               // 타겟 이름
    const [targetImage, setTargetImage] = useState('');                             // 타겟 이미지
    const [previewURL, setPreviewURL] = useState('');                               // 타겟 이미지
    const [target_info, setTarget_info] = useState('');                             // TargetInfo
    const [target_engine, setTarget_engine] = useState('');                         // Engine
    const [target_os, setTarget_os] = useState('');                                 // OS
    const [target_cpu, setTarget_cpu] = useState('');                               // CPU
    const [target_acc, setTarget_acc] = useState('');                               // Accelerator
    const [target_memory, setTarget_memory] = useState('');                         // Memory
    const [target_host_ip, setTarget_host_ip] = useState('');                       // IP Address
    const [target_host_port, setTarget_host_port] = useState('');                   // Port
    const [target_host_service_port, setTarget_host_service_port] = useState('');   // Service Port

    useEffect( () => {

        // 프로젝트 정보 수신
        Request.requestProjectInfo(project_id).then(result =>
        {
            projectContentUpdate(result.data);

            // 타이머 기능
//            if(result.data['container_status'] === 'running')
//            {
//                startTimer()
//            }

            // project_info.yaml 파일을 생성하지 않은 경우
            if(result.data['container'] === '')
            {
                setPannel(true)
            }

            // 데이터 셋 정보 수신
            get_dataset_list(result.data['dataset_list']);
        })
        .catch(error =>
        {
            console.log('project info get error')
        });

        // 타겟 정보 수신
        get_target_list();

        // unmount
        return() => {
            //console.log('unmount')
            clearInterval(timerRef.current)
        }
    }, []);

    // 프로젝트 정보 업데이트
    const projectContentUpdate = (data) =>
    {
        setTarget(parseInt(data['target']))  // 선택 타겟 정보
        setDataset(data['dataset'])          // 데이터 셋 정보
        setTaskType(data['task_type'])
        setNasType(data['nas_type'])

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

        const projectData = {
            'task_type': data['task_type'],
            'target_id': data['target'],
            'nas_type': data['nas_type'],
        }
        setOriData(projectData)
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

    // 타겟 정보 수신
    const get_target_list = () =>
    {
        RequestTarget.requestTargetList().then(result =>
        {
            setTarget_list(result.data);
        })
        .catch(error =>
        {
            console.log('target list get error')
        });
    }

    // 데이터 셋 정보 수신 - 추후 레이블링 저작도구 연동
    const get_dataset_list = (param) =>
    {
        setDataset_list(param)
    }

    const get_target_info = (id) =>
    {
        const findIndex = target_list.findIndex(v => v.id === id)

        if (findIndex !== -1)
        {
            return target_list[findIndex].info
        }
    }

    // 데이터 셋 선택
    const dataSetClick = (value, index) =>
    {
        setDataset(value);
        const dataset_item_box = document.getElementsByClassName("dataset_item_box");

        for (var i=0; i < dataset_item_box.length; i++)
        {
            if (i === index)
            {
                dataset_item_box[i].className = 'dataset_item_box tooltip select';
            }
            else
            {
                dataset_item_box[i].className = 'dataset_item_box tooltip';
            }
        }
    }

    // 데이터 셋 이미지 가져오기
    const getDataset_image = (value) =>
    {
        if (value.indexOf('COCO') !== -1)
        {
            return data_th_3;
        }
        else if (value.indexOf('칫솔') !== -1)
        {
            return data_th_1;
        }
        else if (value.indexOf('파이프') !== -1)
        {
            return data_th_2;
        }
        else if (value.indexOf('폐') !== -1)
        {
            return data_th_4;
        }
        else
        {
            return "";
        }
    };

    // 타겟 선택 이벤트
    const targetClick = (selectTarget, index) =>
    {
       setTarget(selectTarget.id)

       const target_item_box = document.getElementsByClassName("target_item_box");

       for (var i=0; i < target_item_box.length; i++)
       {
           if (i === index)
           {
               target_item_box[i].className = 'target_item_box tooltip select';
           }
           else
           {
               target_item_box[i].className = 'target_item_box tooltip';
           }
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

    // 타겟 생성 입력 폼 정보 초기화
    const stateInfoInit = () =>
    {
        setTargetName('')
        setTargetImage('')
        setPreviewURL('')
        setTarget_info('')
        setTarget_engine('')
        setTarget_os('')
        setTarget_cpu('')
        setTarget_acc('')
        setTarget_memory('')
        setTarget_host_ip('')
        setTarget_host_port('')
        setTarget_host_service_port('')
    };

    // 타겟 추가 버튼 클릭
    const target_add_button_click = () =>
    {
       document.getElementById('create_target_popup').style.display = 'block';
    }

    // 타겟 이미지 업로드
    const uploadImageFile = (event) =>
    {
        const file = event.target.files[0];

        const reader = new FileReader();

        reader.onload = function() {
            setTargetImage(reader.result);

            setPreviewURL(reader.result)
        }

        reader.readAsDataURL(file);

        event.target.value = "";
    };

    // 타겟 수정 팝업 취소 버튼 클릭
    const target_popup_Cancel_ButtonClick = () => {
        stateInfoInit();
        document.getElementById('create_target_popup').style.display = 'none';
    };

    /* 타겟 생성 팝업 - 생성 버튼 클릭 */
    const target_popup_Create_ButtonClick = () =>
    {
        const param = {
            'name': targetName,
            'image': targetImage,
            'info': target_info,
            'engine': target_engine,
            'os': target_os.value,
            'cpu': target_cpu.value,
            'acc': target_acc.value,
            'memory': target_memory,
            'host_ip': target_host_ip,
            'host_port': target_host_port,
            'host_service_port': target_host_service_port
        };

        RequestTarget.requestTargetCreate(param).then(result =>
        {
            document.getElementById('create_target_popup').style.display = 'none';      // 생성 팝업 숨기기

            stateInfoInit();

            get_target_list();
        })
        .catch(error =>
        {
            console.log('description modify error')
        });
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
            if(result === true)
            {
                neuralCreate(param)
            }
        }
    };

    // 신경망 생성 시작
    const neuralCreate = (param) =>
    {
        // 데이터베이스 업데이트
        Request.requestProjectUpdate(param).then(result =>
        {
            //console.log(result)
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

        if(oriData['nas_type'] === 'neck_nas')
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

        const target_id = oriData['target_id'];
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

    // Run Image :  port
    const runImageButtonClick = () =>
    {
        //console.log("runImageButtonClick")

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

        // console.log("containerStart")

        const cookies = new Cookies();
        var user = cookies.get('userinfo')

        RequestContainer.requestContainerStart(name, user, project_id).then(result =>
        {
            // console.log(result)

            status_result_update(JSON.parse(result))
        })
        .catch(error =>
        {
            // console.log(error);

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
        {/* 타겟 생성 팝업 */}
        <TargetPopup.TargetCreatePopup
            popup_modify_mode={false}
            targetName={targetName}
            setTargetName={setTargetName}
            previewURL={previewURL}
            targetImage={targetImage}

            target_info={target_info} setTarget_info={setTarget_info}
            target_engine={target_engine} setTarget_engine={setTarget_engine}
            target_os={target_os} setTarget_os={setTarget_os}
            target_cpu={target_cpu} setTarget_cpu={setTarget_cpu}
            target_acc={target_acc} setTarget_acc={setTarget_acc}
            target_memory={target_memory} setTarget_memory={setTarget_memory}
            target_host_ip={target_host_ip} setTarget_host_ip={setTarget_host_ip}
            target_host_port={target_host_port} setTarget_host_port={setTarget_host_port}
            target_host_service_port={target_host_service_port} setTarget_host_service_port={setTarget_host_service_port}

            uploadImageFile={uploadImageFile}
            cancel_ButtonClick={() => target_popup_Cancel_ButtonClick()}
            create_ButtonClick={target_popup_Create_ButtonClick}/>

        {/* 프로젝트 생성 - 신경망 생성 폼 */}
        <div className='project_manage_container'>

            <div className='project_manage_content' style={{width:'100%'}}>

                <div id='project_top' className='project_top'  style={{padding:'0px 0px 0px 0px', height:'100%'}}>

                <div id="accordion" className="accordion" onClick={ ()=> accordionButtonClick() } style={{height:'40px', position:'static', backgroundColor:'#303030', borderRadius:pannel === false? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>
                    <span style={{fontSize:'16px', color:'white'}}>Information </span>
                    <div id="overall_icon" className="overall_icon" style={{backgroundImage:pannel === false ? "url('" + overall_down + "')" : "url('" + overall_up + "')"}}></div>
                </div>

                <Collapse isOpened={pannel}>
                    <div className="project_user_requirement" style={{display:'flex', maxHeight:'220px'}}>

                        {/* 데이터셋 선택 */}
                        <div className='project_user_requirement_left' style={{ padding:'0px 0px 0px 0px', marginRight:'0px'}}>

                            <div className='select_dataset' style={{padding:'0px 0px 0px 20px', color:'black', display:'flex', alignItems:'center', height:'50px'}}>
                                <div style={{fontSize:'16px', width:'auto'}}>Dataset</div>
                            </div>

                            <div className='dataset_content'>

                                { dataset_list.length > 0 ?
                                    <>
                                    <div className='dataset_list' style={{height:'100%', width:'100%', padding:'0px 20px 20px 20px', backgroundColor:'white', maxHeight:'160px', overflow:'auto'}}>
                                        {dataset_list.map((menu, index) => {
                                            return (
                                             <div key={index} className={dataset === menu ? "dataset_item_box tooltip select" : "dataset_item_box tooltip"} onClick={ ()=> dataSetClick(menu, index)}>
                                                <img id="dataset_item_image" className="dataset_item_image" src={getDataset_image(menu)} style={{height:'100%', width:'100%', margin:'auto', marginRight:'5px', backgroundColor:'#DEDEDE'}}/>
                                                <span className="dataset_tooltiptext" style={{width:'150px'}}>{menu}</span>
                                              </div>
                                            )
                                        })}
                                    </div>
                                    </>
                                    :
                                    <>
                                    <div style={{height:'100%', width:'100%',
                                        fontSize:"50px", fontWeight:'700', textAlign:'center',
                                        display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                                        Please Create DataSet!
                                    </div>
                                    </>
                                }
                            </div>

                        </div>

                        {/* 타겟 선택 */}
                        <div className='project_user_requirement_right' style={{padding:'0px 20px 0px 0px', marginLeft:'0px'}}>
                            <div className='select_target' style={{padding:'0px 0px 0px 20px', color:'black', display:'flex', alignItems:'center', height:'50px', width:'100%'}}>
                                <div style={{ fontSize:'16px',  height:'50%', width:'70%'}}>Target</div>
                                {/* 타겟 생성 버튼 */}
                                <div style={{ width:'30%', height:'70%'}}>
                                    <button onClick={() => target_add_button_click()} style={{height:'100%', width:'100%', borderRadius:'5px', backgroundColor:'#707070', color:'white', fontSize:'0.9em', border:'0px'}}>New Target</button>
                                </div>
                            </div>

                            <div className='dataset_content' style={{display:'block', overflow:'auto'}}>

                                { target_list.length > 0 ?
                                    <>
                                    <div className='dataset_list' style={{height:'100%', width:'100%', padding:'0px 20px 20px 20px', backgroundColor:'white',  maxHeight:'160px', overflow:'auto'}}>
                                        {target_list.map((menu, index) => {
                                            return (
                                              <div className={target === menu.id ? "target_item_box tooltip select" : "target_item_box tooltip"} key={index} onClick={ ()=> targetClick(menu, index)}>
                                                <img id="dataset_item_image" className="dataset_item_image" src={menu.image} style={{height:'100%', width:'100%', margin:'auto', marginRight:'5px', backgroundColor:'#DEDEDE'}}/>
                                                <span className="dataset_tooltiptext" style={{width:'150px'}}>{menu.name}</span>
                                              </div>
                                            )
                                        })}

                                    </div>
                                    </>
                                    :
                                    <>
                                    <div style={{height:'100%', width:'100%',
                                        fontSize:"50px", fontWeight:'700', textAlign:'center',
                                        display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                                        Does Not Exist Targets!
                                    </div>
                                    </>
                                }
                            </div>
                        </div>
                    </div>
                </Collapse>


                <div id="accordion" className="accordion" onClick={ ()=> configAccordionButtonClick() } style={{height:'40px', marginTop:'20px', position:'static', backgroundColor:'#303030', borderRadius:config_pannel === false? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>
                    <span style={{fontSize:'16px', color:'white'}}>Configuration </span>
                    <div id="config_overall_icon" className="config_overall_icon" style={{backgroundImage:config_pannel === false ? "url('" + overall_down + "')" : "url('" + overall_up + "')"}}></div>
                </div>

                <Collapse isOpened={config_pannel}>
                    <div className="project_user_requirement" style={{borderRadius:'0px 0px 5px 5px', border:'5px solid #303030'}}>

                        {/* Task 선택 */}
                        <div className='project_user_requirement_task_type' style={{height:'auto', borderBottom:'3px solid #303030'}}>
                            <div style={{display:"flex", width:'100%', height:'100%'}}>
                                <div style={{width:'20%', minWidth:'150px', backgroundColor:'#707070', textAlign:'center', padding:'10px 10px 10px 10px'}}>
                                    <div style={{padding:'0px 20px 0px 20px', color:'white'}}>Task Type</div>
                                </div>

                                <div style={{width:'80%', display:'flex', padding:'10px 10px 10px 10px'}}>
                                    <input type="radio" name="task_type_radio" value="classification" onChange={({ target: { value } }) => setTaskType(value)} style={{marginLeft:'20px'}} checked={taskType === 'classification'}/><span style={{fontSize:'16px'}}>Classification</span>
                                    <input type="radio" name="task_type_radio" value="detection" onChange={({ target: { value } }) => setTaskType(value)} style={{marginLeft:'20px'}} checked={taskType === 'detection'}/><span style={{fontSize:'16px'}}>Detection</span>
                                </div>
                            </div>
                        </div>

                        {/* AutoNN Configuration */}
                        <div className='project_user_requirement_autonn_config' style={{height:'auto', borderBottom:'3px solid #303030'}}>
                            <div style={{display:"flex", width:'100%', height:'100%'}}>
                                <div style={{width:'20%', minWidth:'150px', backgroundColor:'#707070', textAlign:'center', padding:'15px'}}>
                                    <div style={{alignItems:'center', display:'inline-flex', color:'white'}}>AutoNN Config</div>
                                </div>

                                <div style={{width:'80%', display:'flex', padding:'10px 10px 10px 10px'}}>
                                    <div style={{width:'50%'}}>
                                        <label style={{textAlign:'right', marginLeft:'20px', width:'30%'}}>Dataset file : </label>
                                        <input
                                            className="config-input"
                                            type="text"
                                            placeholder="dataset.yaml"
                                            style={{padding:'0px 10px 0px 10px', width:'60%'}}
                                            maxLength='100'
                                            value={datasetFile} readOnly/>
                                    </div>

                                    <div style={{width:'50%'}}>
                                        <label style={{textAlign:'right', marginLeft:'20px', width:'30%'}}>Base Model : </label>
                                        <input
                                            className="config-input"
                                            type="text"
                                            placeholder="baseModel.yaml"
                                            style={{padding:'0px 10px 0px 10px', width:'60%'}}
                                            maxLength='100'
                                            value={baseModel} readOnly/>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* NAS Type */}
                        <div className='project_user_requirement_nas_type' style={{height:'auto', borderBottom:'3px solid #303030'}}>
                            <div style={{display:"flex", width:'100%', height:'100%'}}>
                                <div style={{width:'20%', minWidth:'150px', backgroundColor:'#707070', textAlign:'center', padding:'10px 10px 10px 10px'}}>
                                    <div style={{padding:'0px 20px 0px 20px', color:'white'}}>Nas Type</div>
                                </div>

                                <div style={{width:'80%', display:'flex', padding:'10px 10px 10px 10px'}}>
                                    <input type="radio" name="nas_type_radio" value="bb_nas" onChange={({ target: { value } }) => setNasType(value)} style={{marginLeft:'20px'}} checked={nasType === 'bb_nas'}/><span style={{fontSize:'16px'}}>Backbone Nas</span>
                                    <input type="radio" name="nas_type_radio" value="neck_nas" onChange={({ target: { value } }) => setNasType(value)} style={{marginLeft:'20px'}} checked={nasType === 'neck_nas'}/><span style={{fontSize:'16px'}}>Neck Nas</span>
                                </div>
                            </div>
                        </div>

                        {/* Deploy Configuration */}
                        <div className='project_user_requirement_deploy_config' style={{height:'auto', display:get_target_info(target) === 'pc' || get_target_info(target) === 'cloud' ? 'block' : 'none'}}>
                            <div style={{display:"grid", width:'100%', height:'100%', gridTemplateColumns:'auto 80%', gridTemplateRows:'1fr 1fr'}}>

                                <div style={{gridRow:'1/3', gridColumn:'1/2', minWidth:'150px', backgroundColor:'#707070', textAlign:'center', padding:'10px 10px 10px 10px'}}>
                                    <div style={{padding:'0px 20px 0px 20px', color:'white', alignItems:'center', display:'inline-flex', height:'100%'}}>Deploy Config</div>
                                </div>

                                <div className='deploy-config' style={{gridRow:'1/2', gridColumn:'2/3'}}>
                                    <div style={{width:'100%', display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr', padding:'10px 10px 10px 20px'}}>
                                        <div style={{ gridColumn:'1/2'}}>
                                            <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Light Weight Level </label>
                                            <input
                                                className="config-input"
                                                type="number"
                                                min="0"
                                                max="10"
                                                step="0"
                                                maxLength={10}
                                                style={{padding:'0px 10px 0px 10px', width:'80%'}}
                                                value={weightLevel}
                                                onChange={({ target: { value } }) => setWeightLevel(value)}
                                                onKeyDown={(evt) => evt.key && evt.preventDefault()}/>
                                        </div>

                                        <div style={{ gridColumn:'2/3'}}>
                                            <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Precision Level</label>
                                            <input
                                                className="config-input"
                                                type="number"
                                                min="0"
                                                max="10"
                                                step="0"
                                                maxLength={2}
                                                style={{padding:'0px 10px 0px 10px', width:'80%'}}
                                                value={precisionLevel}
                                                onChange={({ target: { value } }) => setPrecisionLevel(value)}
                                                onKeyDown={(evt) => evt.key && evt.preventDefault()}/>
                                        </div>

                                        <div style={{ gridColumn:'3/4'}}>
                                            <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Processing Lib</label>
                                            <input
                                                className="config-input"
                                                type="text"
                                                style={{padding:'0px 10px 0px 10px', width:'80%'}}
                                                maxLength='100'
                                                value={processingLib} readOnly/>
                                        </div>

                                        <div style={{ gridColumn:'4/5'}}>
                                            <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>User Editing</label>
                                            <Select options={userEditList} isSearchable={false}
                                            value={userEdit}
                                            onChange={setUserEdit}/>
                                        </div>

                                    </div>
                                </div>

                                <div className='deploy-config' style={{gridRow:'2/3', gridColumn:'2/3'}}>
                                    <div style={{width:'100%', display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr', padding:'10px 10px 10px 20px'}}>
                                        <div style={{ gridColumn:'1/2'}}>
                                            <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Input Method</label>
                                            <Select options={inputMethodList} isSearchable={false}
                                            value={inputMethod}
                                            onChange={setInputMethod}/>
                                        </div>

                                        <div style={{ gridColumn:'2/3'}}>
                                            <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Input Data Path</label>
                                            <input
                                                className="config-input"
                                                type="text"
                                                style={{padding:'0px 10px 0px 10px', width:'80%'}}
                                                maxLength='100'
                                                value={inputDataPath} readOnly/>
                                        </div>

                                        <div style={{ gridColumn:'3/4'}}>
                                            <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Output Method</label>
                                            <Select options={outputMethodList} isSearchable={false}
                                            value={outputMethod}
                                            onChange={setOutputMethod}/>
                                        </div>

                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </Collapse>

                <div id='runButtonArea' className='button' style={{marginTop:'20px', textAlign:'center'}}>

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


