import React from "react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/combo_box.css'
import '../../../../CSS/stepProgress.css'
import '../../../../CSS/project_management.css'

import Combobox from "react-widgets/Combobox";

import * as Request from "../../../../service/restProjectApi";
import * as RequestDummy from "../../../../service/restDummyApi";
import * as RequestData from "../../../../service/restDataApi";
import * as RequestTarget from "../../../../service/restTargetApi";

import overall_up from "../../../../images/icons/icon_3x/chevron-up.png";
import overall_down from "../../../../images/icons/icon_3x/chevron-down.png";

import data_th_1 from "../../../../images/thumbnail/data_th_1.PNG";         // 칫솔
import data_th_2 from "../../../../images/thumbnail/data_th_2.PNG";         // 용접 파이프
import data_th_3 from "../../../../images/thumbnail/data_th_3.PNG";         // 실생활
import data_th_4 from "../../../../images/thumbnail/data_th_4.PNG";         // 폐결핵 판독

import * as TargetPopup from "../../../../components/popup/targetPopup";

function NeuralAndLoadPage({project_id, project_name, project_description})
{
    const [proj_id, setProj_id] = useState(project_id);                                             // 프로젝트 ID
    const [proj_description, setProj_description] = useState(project_description);                 // 프로젝트 설명

    const [project_thumb, setProject_thumb] = useState('');                     // 프로젝트 섬네일 이미지

    const [dataSet, setDataSet] = useState('');                                 // 데이터 셋 경로
    const [dataset_list, setDataset_list] = useState([]);                       // 데이터 셋 리스트

    const [target, setTarget] = useState('');                                   // 타겟 변경
    const [target_list, setTarget_list] = useState([]);                         // 타겟 리스트

    const [step, setStep] = useState(0);                                        // 신경망 생성 단계
    const [stepText, setStepText] = useState('');                               // 신경망 생성 단계 상황

    const [panel, setPanel] = useState('block');                                // panel 창 상태
    const [deploy_panel, setDeploy_panel] = useState('block');                  // deploy_panel 창 상태

    const [serverHost, setServerHost] = useState(window.location.hostname);
    const [showIframe, setShowIframe] = useState(false);                        // iframe 표시 여부
    const [currentWorkNum, setCurrentWorkNum] = useState(0);                    // 현재 작업중인 단계
    const [currentWorkHost, setCurrentWorkHost] = useState('');                 // 현재 작업중인 서버 주소


    // 타겟 정보
    const [popup_modify_mode, setPopup_modify_mode] = useState(false);

    const [targetName, setTargetName] = useState('');                   // 타겟 이름
    const [targetImage, setTargetImage] = useState('');                 // 타겟 이미지
    const [previewURL, setPreviewURL] = useState('');                   // 타겟 이미지
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

        Request.requestProjectInfo(project_id).then(result =>
        {
            // DB 정보 수신
            if (result.data['target'] !== null && target === '')
            {
                // 선택 타겟 정보
                setTarget(parseInt(result.data['target']))
            }

            if(result.data['dataset_path'] !== '' && dataSet === '')
            {
                // 데이터 셋 정보
                setDataSet(result.data['dataset_path'])
            }
            setDataset_list(result.data['dataset_list'])

            // 타겟 리스트 이름 변경
//            var target_change_name_list = []
//            for (var i=0; i < result.data['target_list'].length; i++)
//            {
//                target_change_name_list.push(target_name[result.data['target_list'][i]]);
//            }
//            setTarget_list(target_change_name_list);

            // 타겟 정보 수신
            get_target_list();

            // 신경망 생성 단계 프로그래스바 업데이트
            if(result.data['step'] !== 0)
            {
                setStep(result.data['step'])

                setPanel('none')
            }
        })
        .catch(error =>
        {
            console.log('project info get error')
        });
    }, []);

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

    const description_save_button_click = () => {
        const description = proj_description.trim();
        if(description.length > 0)
        {
            const data = {
                id:proj_id,
                description:proj_description
            }

            Request.requestProjectDescriptionModify(data).then(result =>
            {
                alert('project description update complete')
                console.log('description modify success')
            })
            .catch(error =>
            {
                console.log('description modify error')
            });
        }
        else
        {
            alert('프로젝트 설명을 입력해주세요.')
        }
    }


    const dataSetClick = (value, index) =>
    {
        setDataSet(value);
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
//        const target_key = Object.keys(target_name).find(key => target_name[key] === value);
//        setTarget(target_key)                // 사용자 선택 타겟 정보

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
       console.log("target_add_button_click")

       document.getElementById('create_target_popup').style.display = 'block';
    }

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

    /* 생성 팝업 - 생성 버튼 클릭 */
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

    /* Task Type 선택 이벤트 */
    const changeTaskType = (value) => {

        console.log("changeTaskType");

    }

    // 'Run' 버튼 클릭 이벤트
    const runButtonClick = () =>
    {
        if(step === 0)
        {
            setStepText('')
//            createNeuralNetwork();

            // select 창 숨기기
            if(panel === 'block')
            {
                setPanel('none')
                document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_down + "')";
            }
            databaseUpdate(1);
        }
        else
        {
            var result = window.confirm("새로운 신경망을 생성 하시겠습니까?")
            if(result === true)
            {
                setStepText('')
//                createNeuralNetwork();

                // select 창 숨기기
                if(panel === 'block')
                {
                    setPanel('none')
                    document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_down + "')";
                }
                databaseUpdate(1);
            }
        }
    };

    // select 창 보이기 숨기기
    const accordionButtonClick = () =>
    {
        const acc = document.getElementsByClassName("panel");

        if(acc[0].style.display === 'block')
        {
            setPanel('none')
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_down + "')";
        }
        else
        {
            setPanel('block')
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_up + "')";
        }
    }

    const deployAccordionButtonClick = () =>
    {
        const acc = document.getElementsByClassName("deploy_panel");

        if(acc[0].style.display === 'block')
        {
            setPanel('none')
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_down + "')";
        }
        else
        {
            setPanel('block')
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_up + "')";
        }
    }

    // 데이터베이스 업데이트
    const databaseUpdate = (num) =>
    {
        const param = {
            'project_id' : project_id,
            'project_name' : project_name,
            'project_thumb' : project_thumb,
            'selectTarget' : target,
            'dataset_path' : dataSet,
            'step' : num
        }

        // 데이터베이스 업데이트
        Request.requestProjectUpdate(param).then(result =>
        {
            console.log('Complete Database Upload');
        })
        .catch(error =>
        {
            console.log(error);
        });
    }

    // 단계 별 버튼 클릭
    const progress_button_click = (num) =>
    {
        setStepText('');

        if(currentWorkNum === 0)
        {
            document.getElementById('progress_' + num).className = 'stepper-item2 select';

            setShowIframe(true);

            setCurrentWorkNum(num);
        }
        // 현재 작업중인 단계와 다른 버튼을 클린한 경우
        else if(currentWorkNum !== num )
        {
            document.getElementById('progress_' + currentWorkNum).className = 'stepper-item2 non-select';
            document.getElementById('progress_' + num).className = 'stepper-item2 select';

            setCurrentWorkNum(num);
        }
        // 현재 작업중인 단계와 동일한 버튼을 클린한 경우
        else if(currentWorkNum === num)
        {
            document.getElementById('progress_' + num).className = 'stepper-item2 non-select';

            setShowIframe(false);

            setCurrentWorkNum(0);

            return;
        }

        switch(num)
        {
            case 1 :
                baseModelSelect()
                break;
            case 2 :
                createNeuralNetwork()
                break;
            case 3 :
                createRunImageFile()
                break;
            case 4 :
                deployRunImage()
                break;
            case 5 :
                runNeuralNetwork()
                break;
            default:
                break;
        }
    }

    // Base Model Select: 9000 port
    const baseModelSelect = () =>
    {
        setStepText('iframe 내에 base Select Model 서버(serverHost:9000) 표시');
        setCurrentWorkHost('http://' + serverHost + ':9000');
    }

    // 신경망 생성: 9001 port
    const createNeuralNetwork = () =>
    {
        setStepText('iframe 내에 신경망 자동 생성 서버(serverHost:9001) 표시');
        setCurrentWorkHost('http://' + serverHost + ':9001');
    }

    // 실행 이미지 생성: 9002 port
    const createRunImageFile = () =>
    {
        setStepText('iframe 내에 실행 이미지 생성 서버(serverHost:9002) 표시');
        setCurrentWorkHost('http://' + serverHost + ':9002');
    }

    // 실행 이미지 탑재: 9003 port
    const deployRunImage = () =>
    {
        setStepText('iframe 내에 실행 이미지 다운로드 서버(severHost:9003) 표시');
        setCurrentWorkHost('http://' + serverHost + ':9003');
    }

    // 신경망 실행: 9004 port
    const runNeuralNetwork = () =>
    {
        setStepText('iframe 내에 타겟 원격 실행 서버 (serverHost:9004) 표시');
        setCurrentWorkHost('http://' + serverHost + ':9004');
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

            <div className='project_manage_content' >

                <div>

                    <div id="accordion" className="accordion" onClick={ ()=> accordionButtonClick() } style={{height:'40px', backgroundColor:'#303030', borderRadius:panel === 'none'? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>
                        <span style={{fontSize:'16px', color:'white'}}>Information </span>
                        <div id="overall_icon" className="overall_icon" style={{backgroundImage:panel === 'none'? "url('" + overall_down + "')" : "url('" + overall_down + "')"}}></div>
                    </div>

                    <div className="panel" style={{display:panel}}>

                        <div className="project_description" style={{backgroundColor:'#303030'}}>
                            <div className="description-content" style={{ padding:'10px 20px 10px 20px', height:'100%', backgroundColor:'#303030', display:'flex'}}>
                                <span style={{color:'white'}}>Description</span>
                                <input onChange={({ target: { value } }) => setProj_description(value)} value={proj_description} style={{ height:'100%', width:'100%', borderRadius:'5px', marginLeft:'20px', marginRight:'20px', fontSize:'16px'}} />
                                <button onClick={() => description_save_button_click()} style={{ height:'100%', width:'150px', borderRadius:'5px', backgroundColor:'#4A80FF', color:'white', fontSize:'16px', border:'0px'}}>저장</button>
                            </div>
                        </div>

                        <div className="project_user_requirement" style={{display:'flex'}}>

                            {/* 데이터셋 선택 */}
                            <div className='project_user_requirement_left' style={{ padding:'0px 0px 0px 0px', marginRight:'0px'}}>

                                <div className='select_dataset' style={{padding:'0px 0px 0px 20px', color:'black', display:'flex', alignItems:'center', height:'50px'}}>
                                    <div style={{ fontSize:'16px',  height:'70%', width:'80%'}}>Dataset</div>
                                </div>

                                <div className='dataset_content' style={{display:'block', overflow:'auto', paddingBottom:'60px'}}>

                                    { dataset_list.length > 0 ?
                                        <>
                                        <div className='dataset_list' style={{height:'100%', width:'100%', padding:'0px 20px 20px 20px', backgroundColor:'white'}}>
                                            {dataset_list.map((menu, index) => {
                                                return (
                                                 <div key={index} className={dataSet === menu ? "dataset_item_box tooltip select" : "dataset_item_box tooltip"} onClick={ ()=> dataSetClick(menu, index)}>
                                                    <img id="dataset_item_image" className="dataset_item_image" src={getDataset_image(menu)} style={{height:'100%', width:'100%', margin:'auto', marginRight:'5px', backgroundColor:'#DEDEDE'}}/>
                                                    <span className="dataset_tooltiptext" style={{width:'200px'}}>{menu}</span>
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
                                <div className='select_target' style={{padding:'0px 0px 0px 20px', color:'black', display:'flex', alignItems:'center', height:'20%', width:'100%'}}>
                                    <div style={{ fontSize:'16px',  height:'70%', width:'80%'}}>Target</div>
                                    {/* 타겟 생성 버튼 */}

                                    <div style={{ width:'20%', height:'70%'}}>
                                        <button onClick={() => target_add_button_click()} style={{height:'100%', width:'100%', borderRadius:'5px', backgroundColor:'#707070', color:'white', fontSize:'0.9em', border:'0px'}}>New Target</button>
                                    </div>
                                </div>

                                <div className='dataset_content' style={{display:'block', overflow:'auto', paddingBottom:'60px'}}>

                                    { target_list.length > 0 ?
                                        <>
                                        <div className='dataset_list' style={{height:'100%', width:'100%', padding:'0px 20px 20px 20px', backgroundColor:'white'}}>
                                            {target_list.map((menu, index) => {
                                                return (
                                                  <div className={target === menu.id ? "target_item_box tooltip select" : "target_item_box tooltip"} key={index} onClick={ ()=> targetClick(menu, index)}>
                                                    <img id="dataset_item_image" className="dataset_item_image" src={menu.image} style={{height:'100%', width:'100%', margin:'auto', marginRight:'5px', backgroundColor:'#DEDEDE'}}/>
                                                    <span className="dataset_tooltiptext" style={{width:'200px'}}>{menu.name}</span>
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

                        {/* Task 선택 */}
                        <div className='project_user_requirement_task_type' style={{height:'50px', marginLeft:'0px'}}>
                            <div style={{display:"flex", width:'100%', height:'100%'}}>
                                <div style={{width:'20%', backgroundColor:'#303030', textAlign:'center'}}>
                                    <div style={{padding:'0px 20px 0px 20px', color:'white', marginTop:'10px'}}>Task Type</div>
                                </div>

                                <input type="radio" name="task_type_radio" value="classification" onChange={({ target: { value } }) => changeTaskType(value)} style={{marginLeft:'20px'}}/><span style={{marginTop:'10px', fontSize:'16px'}}>Classification</span>
                                <input type="radio" name="task_type_radio" value="detection" onChange={({ target: { value } }) => changeTaskType(value)} style={{marginLeft:'20px'}}/><span style={{marginTop:'10px', fontSize:'16px'}}>Detection</span>
                            </div>
                        </div>

                        {/* AutoNN Configuration */}
                        <div className='project_user_requirement_autonn_config' style={{height:'50px', marginLeft:'0px'}}>
                            <div style={{display:"flex", width:'100%', height:'100%'}}>
                                <div style={{width:'20%', backgroundColor:'#303030', textAlign:'center'}}>
                                    <div style={{padding:'0px 20px 0px 20px', color:'white', marginTop:'10px'}}>AutoNN Config</div>
                                </div>

                                <div style={{width:'80%', display:'flex'}}>
                                    <div style={{width:'50%', margin:'5px 5px 5px 5px'}}>
                                        <label style={{textAlign:'right', marginLeft:'10px', width:'30%'}}>Dataset file : </label>
                                        <input
                                            className="config-input"
                                            type="text"
                                            placeholder="dataset.yaml"
                                            style={{padding:'0px 10px 0px 10px', width:'60%'}}
                                            maxLength='100'
                                            value={target_memory}
                                            onChange={({ target: { value } }) => setTarget_memory(value)} disabled/>
                                    </div>

                                    <div style={{width:'50%', margin:'5px 5px 5px 5px'}}>
                                        <label style={{textAlign:'right', marginLeft:'10px', width:'30%'}}>Base Model : </label>
                                        <input
                                            className="config-input"
                                            type="text"
                                            placeholder="baseModel.yaml"
                                            style={{padding:'0px 10px 0px 10px', width:'60%'}}
                                            maxLength='100'
                                            value={target_memory}
                                            onChange={({ target: { value } }) => setTarget_memory(value)}/>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div id='runButtonArea' style={{marginTop:'10px', textAlign:'center'}}>
                            { target !== '' && dataSet !== '' ?
                                <button onClick={ ()=> runButtonClick() } style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#4A80FF', color:'white'}}>신경망 자동 생성</button>
                            :
                                <button style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#707070', color:'white'}} readOnly>신경망 자동 생성</button>
                            }
                        </div>
                    </div>
                </div>


                <div id="accordion" className="accordion" onClick={ ()=> deployAccordionButtonClick() } style={{height:'40px', backgroundColor:'#303030', borderRadius:deploy_panel === 'none'? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>
                    <span style={{fontSize:'16px', color:'white'}}>Deploy Config Optional </span>
                    <div id="overall_icon" className="overall_icon" style={{backgroundImage:deploy_panel === 'none'? "url('" + overall_down + "')" : "url('" + overall_down + "')"}}></div>
                </div>

                <div className="deploy_panel" style={{display:deploy_panel}}>
                    {/* Deployment Configuration */}
                    <div className='project_user_requirement_deploy_config' style={{height:'auto', marginLeft:'0px'}}>
                        <div style={{display:"flex", width:'100%', height:'100%'}}>
                            <div style={{width:'20%', backgroundColor:'#303030', textAlign:'center'}}>
                                <div style={{padding:'0px 20px 0px 20px', color:'white', marginTop:'10px'}}></div>
                            </div>

                        </div>
                    </div>
                </div>

                {/*
                <div id='project_bottom' className='project_bottom'  style={{padding:'20px 0px 0px 0px', height:'100%',marginBottom:'0px', gridRow: panel === 'none' ? '2/4' : '3/4' }}>
                    <div className='create_neural_network' style={{ backgroundColor:'#303030', borderRadius:'5px', height:'100%', padding:'10px 20px 20px 20px'}}>
                        <div style={{marginBottom:'10px', display:'flex'}}>
                            <span style={{fontSize:'16px', color:'white'}}>Current Work - </span>
                            <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}>[ </span>
                            <span style={{color:'#4A80FF'}}>{stepText}</span>
                            <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}> ]</span>
                        </div>

                        <div className='status_level' style={{backgroundColor:'white', padding:'20px 0px', borderRadius:'5px'}}>
                            <div className="stepper-wrapper2" id='progressbar'>
                                <div className="stepper-item2 non-select" id='progress_1'>
                                    <div className="step-counter2" onClick={() => progress_button_click(1)}>Base Model Select</div>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_2'>
                                    <div className="step-counter2" onClick={() => progress_button_click(2)}>신경망 자동 생성</div>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_3'>
                                    <div className="step-counter2" onClick={() => progress_button_click(3)}>실행 이미지 생성</div>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_4'>
                                    <div className="step-counter2" onClick={() => progress_button_click(4)}>실행 이미지 다운로드</div>
                                </div>
                                <div className="stepper-item2 non-select" id='progress_5'>
                                    <div className="step-counter2" onClick={() => progress_button_click(5)}>타겟 원격 실행</div>
                                </div>
                            </div>
                        </div>

                        <div className='status_log' style={{color:'white', height:'auto', overflow:'auto',  padding:'20px 0px 0px 0px'}}>
                            <div style={{ border:'2px solid white', borderRadius:'5px', backgroundColor:'white', position:'relative', width:'100%', height:'100%'}}>
                                {/* 각 버튼에 해당하는 서버별
                                {showIframe === true &&
                                    <iframe id='iframe' title='iframe' src={currentWorkHost} frameBorder='0' style={{ width:'100%', height:'100%'}}></iframe>
                                }
                            </div>
                        </div>
                    </div>
                </div>
                */}
            </div>
        </div>
        </>
    );
}

export default NeuralAndLoadPage;


