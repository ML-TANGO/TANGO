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
import WorkFlowForm from "./workFlowForm";

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

    let test_list = [];

    const timerRef = useRef();

    const [yamlMake, setYamlMake] = useState(false);                   // project_info.yaml 생성 여부
    const [container, setContainer] = useState('');                    // 신경망 생성 단계
    const [container_status, setContainer_status] = useState('');      // 신경망 생성 단계 상황

    const [panel, setPanel] = useState(false);                       // panel 창 상태

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
        // 프로젝트 정보 조회
        get_project_info(project_id)

    }, []);

    // 프로젝트 정보 조회
    const get_project_info = (project_id) =>
    {
        // 프로젝트 정보 수신
        Request.requestProjectInfo(project_id).then(result =>
        {
            // 프로젝트 정보 업데이트
            projectContentUpdate(result.data);

            // project_info.yaml 파일을 생성 하지 않은 경우
            if(result.data['yaml_make'] === false)
            {
                setPanel(true)
            }
        })
        .catch(error =>
        {
            console.log('project info get error')
        });
    }

    // 프로젝트 정보 업데이트
    const projectContentUpdate = (data) =>
    {
        console.log('projectContentUpdate')

        console.log(data)

        setTarget(data['target'] !== '' ? data['target'] : '')  // 선택 타겟 정보

        if(data['dataset'] !== ''){
            setDataset(data['dataset'])         // 데이터 셋 정보
            setTaskType(data['task_type'])
        }

        setNasType(data['nas_type'] !== '' ? data['nas_type'] : '')
        setWeightLevel( data['deploy_weight_level'] !== '' ? parseInt(data['deploy_weight_level']) : 0)
        setPrecisionLevel(data['deploy_precision_level'] !== '' ? parseInt(data['deploy_precision_level']) : 0)

        const im_index = inputMethodList.findIndex(im => im.value === data['deploy_input_method'])
        setInputMethod(im_index !== -1 ? inputMethodList[im_index] : '')

        const om_index = outputMethodList.findIndex(om => om.value === data['deploy_output_method'])
        setOutputMethod(om_index !== -1 ? outputMethodList[om_index] : '')

        const ue_index = userEditList.findIndex(ue => ue.value === data['deploy_user_edit'])
        setUserEdit(ue_index !== -1 ? userEditList[ue_index] : '')

        setYamlMake(data['yaml_make'])                  // project_info.yaml 파일 생성 여부
        setContainer(data['container'])                 // 진행중 컨테이너
        setContainer_status(data['container_status'])   // 진행중 컨테이너 상태
    }

    // Information 창 보이기 숨기기
    const accordionButtonClick = () =>
    {
        if(panel === true)
        {
            setPanel(false)
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_down + "')";
        }
        else{
            setPanel(true)
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_up + "')";
        }
    }

    // 신경망 생성 정보 저장
    const neuralInfoSaveClick = () =>
    {
        const param = {
            'project_id' : project_id,
            'project_target' : target.id,
            'project_dataset' : dataset.DATASET_CD,
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

        if(yamlMake === false)
        {
            neuralInfoSave(param)
        }
        else
        {
            var result = window.confirm("신경망 생성 정보를 수정하시겠습니까?")
            if(result === true) neuralInfoSave(param)
        }
    };

    // 신경망 생성 정보 저장
    const neuralInfoSave = (param) =>
    {
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
                console.log('project update error')
            });
        })
        .catch(error =>
        {
            console.log(error);
        });
    }

    // 신경망 수동 생성 버튼 클릭
    const neuralCreateManualClick = () =>
    {
        if (yamlMake === false)
        {
            alert('신경망 생성 정보를 저장해주세요.')
            return;
        }
        else
        {
            console.log('neuralCreateManualClick')

            // TODO : 컨테이너 상태 업데이트 - bms
            // TODO : BMS 버튼 활성화
        }

        if(yamlMake === true)
        {
            var result = window.confirm("신경망 생성을 다시 시작하시겠습니까?")
            if(result === true)
            {
                // TODO : 컨테이너 상태 업데이트 - bms
                // TODO : BMS 버튼 활성화
            }
        }
    }

    // 신경망 자동 생성
    const neuralCreateAutoClick = () =>
    {
        alert('서비스 준비중입니다.')
    }

    return (
        <>

        {/* 프로젝트 생성 - 신경망 생성 폼 */}
        <div className='project_manage_container'>
            <div className='project_manage_content' style={{width:'100%'}}>
                <div id='project_top' className='project_top'  style={{padding:'0px 0px 0px 0px', height:'100%'}}>

                    <div id="accordion" className="accordion" onClick={ ()=> accordionButtonClick() }
                        style={{height:'40px', position:'static', backgroundColor:'#303030', borderRadius:panel === false? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>

                        <span style={{fontSize:'16px', color:'white'}}>Information </span>
                        <div id="overall_icon" className="overall_icon" style={{backgroundImage:panel === false ? "url('" + overall_down + "')" : "url('" + overall_up + "')"}}></div>
                    </div>

                    <Collapse isOpened={panel}>
                        <InformationForm
                            dataset={dataset} setDataset={setDataset}
                            dataset_list={dataset_list} setDataset_list={setDataset_list}
                            target={target} setTarget={setTarget}
                            target_list={target_list} setTarget_list={setTarget_list}
                            setTaskType={setTaskType}/>

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
                            outputMethod={outputMethod} setOutputMethod={setOutputMethod}/>

                        <div id='neuralSave' className='neuralSave' style={{padding:'10px', textAlign:'center'}}>
                            {target.info === 'pc' || target.info === 'cloud' ?
                                <>
                                { target !== '' && dataset !== '' && taskType !== '' && nasType !== '' && userEdit !== '' && inputMethod !== '' && outputMethod !== ''?
                                    <button onClick={ ()=> neuralInfoSaveClick() } style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor: '#4A80FF', color:'white'}}>신경망 생성 정보 저장</button>
                                :
                                    <button style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#707070', color:'white'}} readOnly>신경망 생성 정보 저장</button>
                                }
                                </>
                            :
                                <>
                                { target !== '' && dataset !== '' && taskType !== '' && nasType !== ''?
                                    <button onClick={ ()=> neuralInfoSaveClick() } style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#4A80FF', color:'white'}}>신경망 생성 정보 저장</button>
                                :
                                    <button style={{height:'42px', width:'30%', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#707070', color:'white'}} readOnly>신경망 생성 정보 저장</button>
                                }
                                </>
                            }
                        </div>
                    </Collapse>

                    <div id='neuralRun' className='neuralRun' style={{paddingTop:'20px', textAlign:'center'}}>
                        <button className='neuralRunButton' onClick={ ()=> neuralCreateManualClick() } style={{ backgroundColor:(yamlMake === true ? '#4A80FF' : '#707070')}}>수동 생성</button>
                        <button className='neuralRunButton' onClick={ ()=> neuralCreateAutoClick() } style={{marginLeft:'50px', backgroundColor:(yamlMake === true ? '#4A80FF' : '#707070')}}>자동 생성</button>
                    </div>
                </div>

                <WorkFlowForm project_id={project_id} container={container} container_status={container_status}/>
            </div>
        </div>
        </>
    );
}

export default NeuralAndLoadPage;
