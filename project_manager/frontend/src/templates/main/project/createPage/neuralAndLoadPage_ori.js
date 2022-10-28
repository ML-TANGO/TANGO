import React from "react";
import { useEffect, useState } from "react";

import { useLocation } from "react-router-dom";

import '../../../../CSS/project_management.css'
import '../../../../CSS/stepProgress.css'

import * as Request from "../../../../service/restProjectApi";
import * as RequestDummy from "../../../../service/restDummyApi";
import * as RequestData from "../../../../service/restDataApi";

import Combobox from "react-widgets/Combobox";
//import "react-widgets/styles.css";
import '../../../../CSS/combo_box.css'

var target_sw_info = {};        // 타겟 SW 정보

var targetYamlPath = '';        // target_yaml 파일 경로
var dataYamlPath = '';          // dataset_yaml 파일 경로

var neural_model_path = '';     // 신경망 모델 경로
var neural_model_name = '';     // 신경망 모델 이름

var source_image_path = '';     // 소스 이미지 경로
var target_image_path = '';     // 타겟 이미지 저장 경로

//var server_ip = '';             // 서버 IP 주소 - 타 서버 요청시 필요

function NeuralAndLoadPage()
{
    const {state} = useLocation();                                              // 상위 페이지 state 정보 - [ 프로젝트 id, 프로젝트 이름 ]

//    const [server_ip, setServer_ip] = useState();                             // 서버 IP 주소 - 타 서버 요청시 필요

    const [target, setTarget] = useState(0);                                    // 타겟 변경

    const [data_path, setData_path] = useState('');                             // 데이터 셋 경로
    const [dataset_list, setDataset_list] = useState([]);                       // 데이터 셋 리스트
    const [data_path_check, setData_path_check] = useState(false);              // 데이터 셋 경로 유무
    const [runButtonActive, setRunButtonActive] = useState(false);              // 'Run' 버튼 활성화 상태

    const [yamlCheck, setYamlCheck] = useState(false);                          // yaml file 유효성 검사 결과
    const [rawDataCheck, setRawDataCheck] = useState(false);                    // raw data 유효성 검사 결과
    const [annotationDataCheck, setAnnotationDataCheck] = useState(false);      // annotation data 유효성 검사 결과
    const [trainDataCheck, setTrainDataCheck] = useState(false);                // train data 유효성 검사 결과
    const [valDataCheck, setValDataCheck] = useState(false);                    // validation data 유효성 검사 결과

    const [step, setStep] = useState(0);                                        // 신경망 생성 단계


    useEffect( () => {

        Request.requestProjectInfo(state.id).then(result =>
        {
//            console.log('DB 정보 수신')
            // DB 정보 수신
            if (result.data['target'] !== 0 && target === 0)
            {
                setTarget(result.data['target'])                        // 선택 타겟 정보

                document.getElementById('target_' + result.data['target']).className = 'selectTarget';

                targetYamlPath = result.data['target_yaml_path']

                target_sw_info['os'] = result.data['os']
                target_sw_info['mllib'] = result.data['mllib']
                target_sw_info['accel'] = result.data['accel']
                target_sw_info['dep'] = result.data['dep']
            }

            if(result.data['dataset_path'] !== '' && data_path === '')
            {
                setData_path(result.data['dataset_path'])               // 데이터 셋 경로
            }
            setDataset_list(result.data['dataset_list'])


            // 신경망 생성 단계 프로그래스 바 업데이트
            if(result.data['step'] !== 0)
            {
                setStep(result.data['step'])                            // 신경망 생성 단계

                step_progress_bar_update(result.data['step'], false)
            }
        })
        .catch(error =>
        {
            alert('project info get error')
        });
    }, []);

        // 단계 프로그래스바 업데이트
    const step_progress_bar_update = (current_step, b_step_status) =>
    {
        var progress_bar = document.getElementById('progressbar');

        for (var i = 1; i < progress_bar.childNodes.length + 1; i++)
        {
            if( i < current_step )
            {
                document.getElementById('progress_' + i).className = 'complete'
            }
            else if( i === current_step )
            {
                if(b_step_status === false)
                {
                    document.getElementById('progress_' + i).className = 'ready'
                }
                else
                {
                    document.getElementById('progress_' + i).className = 'active'
                }
            }
            else
            {
                document.getElementById('progress_' + i).className = 'before'
            }
        }
    }

    // 타겟 선택 이벤트
    const targetChange = (num) =>
    {
        setTarget(num)              // 사용자 선택 타겟 정보 변경

        setData_path_check(false)   // 데이터 셋 경로 확인 초기화
        setRunButtonActive(false)   // 'Run' 버튼 비활성화

        // 선택한 항목을 제외한 버튼들의 클래스 이름 변경
        var buttonList = document.getElementsByClassName('target_item')
        for (var i=1; i<buttonList.length + 1; i++)
        {
            if(i !== num)
            {
                document.getElementById('target_' + i).className = 'nonSelectTarget';
            }
            else
            {
                document.getElementById('target_' + i).className = 'selectTarget';
            }
        }

        const param = {
            'selectTarget' : num,
        }

        // target.yaml 생성
        Request.requestCreateTargetYaml(param).then(result =>
        {
            // target yaml 경로 받아오기
            targetYamlPath = result.data['target_yaml_path']

            // 타겟 SW 정보
            target_sw_info['os'] = result.data['os']
            target_sw_info['mllib'] = result.data['mllib']
            target_sw_info['accel'] = result.data['accel']
            target_sw_info['dep'] = result.data['dep']
        })
        .catch(error =>
        {
            alert('Crate Target Yaml error')
        });
    }

    // 데이터 셋 - 경로 Input 태그 값 변경 시 호출 이벤트
    const datasetPath_update = (value) =>
    {
        setData_path(value)         // 데이터 셋 경로 변경
        setData_path_check(false)   // 데이터 셋 경로 확인 초기화
        setRunButtonActive(false)   // 'Run' 버튼 비활성화
    };


    // 데이터 셋 - 유효성 검사
    const showDatasetListClick = () =>
    {
        if(data_path.trim().length > 0)
        {
            Request.requestDataSetAvailabilityCheck(data_path).then(result =>
            {
                // 파일 경로가 존재하는 경우
                if(result['isPath'] === true)
                {
                    setYamlCheck(result['yaml_file'])
                    setRawDataCheck(result['raw_data'])
                    setAnnotationDataCheck(result['annotation_data'])
                    setTrainDataCheck(result['train_data'])
                    setValDataCheck(result['val_data'])

                    setData_path_check(true)

                    // 3 가지 항목 모두 true인 경우
                    if (result['raw_data'] === true && result['annotation_data'] === true && result['yaml_file'] === true && result['train_data'] === true && result['val_data'] === true )
                    {
                        setRunButtonActive(true)
                        dataYamlPath = result['yaml_file_path'];
                    }
                    else
                    {
                        setRunButtonActive(false)
                        dataYamlPath = '';
                    }
                }
                else
                {
                    setData_path_check(false)
                    alert('데이터 셋이 존재하지 않습니다')
                }
            })
            .catch(error =>
            {
                alert('dataset info confirm error')
            });
        }
        else
        {
            alert('데이터 셋을 입력해주세요')
        }
    };

    // 데이터베이스 업데이트
    const databaseUpdate = (num) =>
    {
        const param = {
            'project_id' : state.id,
            'project_name' : state.name,
            'selectTarget' : target,
            'dataset_path' : data_path,
            'step' : num,
            'targetYamlPath' : targetYamlPath,
            'dataYamlPath' : dataYamlPath,
            'neural_model_path' : neural_model_path,
        }

        // 데이터베이스 업데이트
        Request.requestProjectUpdate(param).then(result =>
        {
            setStep(num)                  // 신경망 생성 단계로 초기화

            step_progress_bar_update(num, false)
        })
        .catch(error =>
        {
            alert('project update error')
        });
    }

    // 'Run' 버튼 클릭 시 이벤트
    const runButtonClick = () =>
    {
        if(step === 0)
        {
            project_init()
        }
        else
        {
            var result = window.confirm("새로운 신경망을 생성 하시겠습니까?")
            if(result === true)
            {
                project_init()
            }
        }
    };

    // 신경망 프로세스 초기화
    const project_init = () =>
    {
        databaseUpdate(1)

        createNeuralNetwork()

        setData_path_check(false)
        setRunButtonActive(false)
    }

    // 단계 별 버튼 클릭
    const progress_button_click = (num) =>
    {
        // 현재 활성화 된 버튼만 클리이 가능하도록 조건 추가
        if(num === step)
        {
            switch(num)
            {
                case 1 :
                    createNeuralNetwork()
                    break;
                case 2 :
                    createRunImageFile()
                    break;
                case 3 :
                    deployRunImage()
                    break;
                case 4 :
                    runNeuralNetwork()
                    break;
            }
        }
        else if (num < step)
        {
            alert('완료된 항목입니다.')
        }
        else
        {
            alert('비활성화 항목입니다.')
        }
    }

    // 신경망 생성
    const createNeuralNetwork = () =>
    {
        status_result_update('신경망 생성 - 시작')
        step_progress_bar_update(1, true)

        const param = {
            'data_yaml_path': dataYamlPath,
            'target_yaml_path': targetYamlPath
        }

        // 신경망 생성 요청
        RequestDummy.requestCreateNeural_Dummy(param).then(result =>
        {
            neural_model_name = result.data['neural_model_name']
            neural_model_path = result.data['neural_model_path']

            // 신경망 생성 완료
            databaseUpdate(2)

            status_result_update('신경망 생성 - 완료 : 신경망 모델 경로 수신')
        })
        .catch(error =>
        {
            step_progress_bar_update(1, false)
            alert('Crate Neural Network error')

            console.log(error)
        });
    }

    // 실행 이미지 생성
    const createRunImageFile = () =>
    {
        status_result_update('실행 이미지 생성 - 시작')
        step_progress_bar_update(2, true)

        const param = {
            'source_image_path': '소스 이미지 경로',
            'target_image_save_path': '타겟 이미지 저장 경로',
            'target_name': target,
            'target_os' : target_sw_info['os'],
            'target_engine' : target_sw_info['accel'],
            'target_ml_lib' : target_sw_info['mllib'],
            'target_module' : target_sw_info['dep'],
            'neural_model_save_path' : neural_model_path,
            'neural_run_app_path' : '신경망 실행 app 저장 경로',
        }

        // 실행 이미지 생성
        RequestDummy.requestCreateImage_Dummy(param).then(result =>
        {
            // 타겟 이미지 저장 경로
            target_image_path = result.data['target_image_save_path']

            // 실행 이미지 생성 완료
            databaseUpdate(3)

            status_result_update('실행 이미지 생성 - 완료 : 타겟 이미지 저장 경로 수신')
        })
        .catch(error =>
        {
            step_progress_bar_update(2, false)
            alert('Image File Make error')

            console.log(error)
        });
    }

    // 실행 이미지 탑재
    const deployRunImage = () =>
    {
        status_result_update('실행 이미지 탑재 - 시작')

        step_progress_bar_update(3, true)

        const param = {
            'target_image_save_path': target_image_path,
            'target_url': '타겟 URL',
            'startup_command' : './rk3399pro-onnx-detector --model=best_detnn-coco-rk3399pro.onnx'
        }

        // 실행 이미지 탑재
        RequestDummy.requestDeployImage_Dummy(param).then(result =>
        {
            // 실행 이미지 탑재 완료
            databaseUpdate(4)

            status_result_update('실행 이미지 탑재 - 완료')
        })
        .catch(error =>
        {
            step_progress_bar_update(3, false)
            alert('Image File Make error')

            console.log(error)
        });
    }

    // 신경망 실행
    const runNeuralNetwork = () =>
    {
       status_result_update('신경망 실행 - 시작')

        step_progress_bar_update(4, true)

        setTimeout(function()
        {
            // 신경망 생성 완료
            databaseUpdate(5)

            status_result_update('신경망 실행 - 완료')

        }, 5000);
    }

    const status_result_update = (text) =>
    {
        document.getElementById('status_result').value += (text + '\n');
    }

    return (
        <>
        {/* 프로젝트 생성 - 신경망 생성 폼 */}
        <div className='manage_container'>
            <div className='project_manage_content'>

                <div className='project_user_requirement'>

                    <div className='project_user_requirement_left' style={{borderRadius:'5px',  padding:'20px 20px 20px 20px', marginRight:'10px'}}>

                        {/* 타겟 선택 */}

                        <div className='select_target' style={{color:'black'}}>
                            <span style={{ fontSize:'16px'}}>Select Target</span>

                             {/*
                            <div className='target_list' style={{marginTop:'15px', display:'flex', height:'100%', width:'auto'}}>
                                <div className='target_item' style={{textAlign:'left'}}>
                                    <button className='nonSelectTarget' id='target_1' type='button' onClick={ () => targetChange(1) } style={{}}><div>온디바이스</div>( RK3399Pro )</button>
                                </div>

                                <div className='target_item' style={{marginLeft:'25px', textAlign:'left'}}>
                                    <button className='nonSelectTarget' id='target_2' type='button' onClick={ () => targetChange(2) } style={{}}><div>온디바이스</div>( Jetson Nano )</button>
                                </div>

                                <div className='target_item' style={{marginLeft:'25px', textAlign:'left'}}>
                                    <button className='nonSelectTarget' id='target_3' type='button' onClick={ () => targetChange(3) } style={{}}><div>엣지클라우드</div>( x86_64 / CUDA )</button>
                                </div>

                                <div className='target_item' style={{marginLeft:'25px', textAlign:'left'}}>
                                    <button className='nonSelectTarget' id='target_4' type='button' onClick={ () => targetChange(4) } style={{}}><div>클라우드</div>( GCP, AWS )</button>
                                </div>
                            </div>
                            */}

                            <div style={{marginTop:'15px'}}>
                                <Combobox placeholder='Select Target' filter={false} data={target_list} />
                            </div>


                        </div>


                    </div>


                    <div className='project_user_requirement_right' style={{borderRadius:'5px',  padding:'20px 20px 20px 20px', marginLeft:'10px'}}>

                        {/* 데이터 셋 선택 */}
                        <div className='select_dataset'>

                            <span style={{fontSize:'16px'}}>Select Dataset</span>

                            <div style={{marginTop:'15px'}}>

                                {/* TODO : 드롭다운 생성 */}
                                {/*
                                <input className='userRequirementInput' id='selectDatasetInput' type='text' placeholder="Input dataset code name  Ex> 'DI00000'" value={data_path} onChange={({ target: { value } }) => datasetPath_update(value)}/>
                                <button className='userRequirementButton' onClick={ ()=> showDatasetListClick() } style={{width:'150px'}}>Availability Check</button>
                                <span id='dataset_check_result' style={{marginLeft:'0px', marginTop:'10px'}}>
                                    { data_path_check === true &&
                                         <>
                                         <span style={{marginLeft:'20px'}}>{datasetFileResult('yaml file', yamlCheck)}</span>
                                         <span style={{marginLeft:'20px'}}>{datasetFileResult('Raw data', rawDataCheck)}</span>
                                         <span style={{marginLeft:'20px'}}>{datasetFileResult('Annotation data', annotationDataCheck)}</span>
                                         <span style={{marginLeft:'20px'}}>{datasetFileResult('train data', trainDataCheck)}</span>
                                         <span style={{marginLeft:'20px'}}>{datasetFileResult('validation data', valDataCheck)}</span>
                                         </>
                                    }
                                </span>
                                */}

                                <span style={{display:'flex'}}>

                                    {/* 데이터 셋 리스트가 없는 경우 */}
                                    { dataset_list.length === 0 ?
                                        <Combobox placeholder='data Set Name...' filter={false} value={data_path} data={['No DataSet list']} disabled={['No DataSet list']} onChange={value => datasetPath_update(value)}/>
                                    :
                                        <Combobox placeholder='data Set Name...' filter={false} value={data_path} data={dataset_list} onChange={value => datasetPath_update(value)}/>
                                    }

                                    {/*
                                    <button className='userRequirementButton' onClick={ ()=> showDatasetListClick() } style={{width:'150px'}}>Availability Check</button>

                                    <span id='dataset_check_result' style={{marginLeft:'0px', marginTop:'10px'}}>
                                        { data_path_check === true &&
                                             <>
                                             <span style={{marginLeft:'20px'}}>{datasetFileResult('yaml file', yamlCheck)}</span>
                                             <span style={{marginLeft:'20px'}}>{datasetFileResult('Raw data', rawDataCheck)}</span>
                                             <span style={{marginLeft:'20px'}}>{datasetFileResult('Annotation data', annotationDataCheck)}</span>
                                             <span style={{marginLeft:'20px'}}>{datasetFileResult('train data', trainDataCheck)}</span>
                                             <span style={{marginLeft:'20px'}}>{datasetFileResult('validation data', valDataCheck)}</span>
                                             </>
                                        }
                                    </span>
                                    */}
                                </span>
                            </div>
                    </div>




                    </div>
                </div>

                <div id='runButtonArea' style={{marginTop:'25px'}}>
                    { target !== 0 && runButtonActive === true ?
                        <button onClick={ ()=> runButtonClick() } style={{height:'42px', width:'150px', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#4A80FF', color:'white'}}>신경망 자동 생성</button>
                    :
                        <button style={{height:'42px', width:'150px', borderRadius:'3px', border:'0', fontSize:'16px', backgroundColor:'#707070', color:'white'}} readOnly>신경망 자동 생성</button>
                    }
                </div>

                <div style={{padding:'20px 0px 0px 0px'}}>

                    <div className='create_neural_network' style={{color:'black', backgroundColor:'#DFDFDF', borderRadius:'5px', height:'100%', padding:'20px 20px 20px 20px'}}>

                        <div className='status_level' style={{backgroundColor:'white', padding:'10px 0px', borderRadius:'5px'}}>
                            <ul className="progressbar" id='progressbar'>
                                <li className="before" id='progress_1' onClick={() => progress_button_click(1)}><div className='list_text' style={{marginTop:'14px'}}>신경망 자동 생성</div></li>
                                <li className="before" id='progress_2' onClick={() => progress_button_click(2)}><div className='list_text' style={{marginTop:'14px'}}>실행 이미지 생성</div></li>
                                <li className="before" id='progress_3' onClick={() => progress_button_click(3)}><div className='list_text' style={{marginTop:'14px'}}>실행 이미지 다운로드</div></li>
                                <li className="before" id='progress_4' onClick={() => progress_button_click(4)}><div className='list_text' style={{marginTop:'14px'}}>타겟 원격 실행</div></li>
                            </ul>
                        </div>

                        <div className='status_log' style={{padding:'20px 0px 0px 0px', display:'grid', gridTemplateRows: '16px 1fr'}}>
                            <span style={{fontSize:'16px'}}>Status Log</span>

                            <div style={{padding:'20px 0px 0px 0px', height:'100%'}}>
                                <textarea id='status_result' style={{backgroundColor:'white', color:'white', resize:'none', width:'100%', height:'100%',borderRadius:'5px', border:'0px'}} readOnly/>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>
        </>
    );
}



// 체크 표시 컴포넌트
function trueCheckSVG()
{
    return <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-check" viewBox="0 0 16 16" style={{verticalAlign:'sub'}}>
                <path d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425a.267.267 0 0 1 .02-.022z"/>
           </svg>
}

// X 표시 컴포넌트
function falseCheckSVG()
{
    return <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" className="bi bi-x" viewBox="0 0 16 16" style={{verticalAlign:'sub'}}>
                <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
            </svg>
}

// 데이터 셋 유효성 거검사 결과 컴포넌트 - [ yaml file, raw data, annotation data ]
function datasetFileResult(text, value)
{
    if(value === true)
    {
        return <span className='dataset_availability_true'>{text} {trueCheckSVG()}</span>
    }
    else
    {
        return <span className='dataset_availability_false'>{text} {falseCheckSVG()}</span>
    }
}

export default NeuralAndLoadPage;


