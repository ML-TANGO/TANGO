import React from "react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/project_management.css'
import '../../../../CSS/stepProgress.css'
//import '../../../../CSS/test.css'

import * as Request from "../../../../service/restProjectApi";
import * as RequestDummy from "../../../../service/restDummyApi";
import * as RequestData from "../../../../service/restDataApi";

import Combobox from "react-widgets/Combobox";
//import "react-widgets/styles.css";
import '../../../../CSS/combo_box.css'

import overall_up from "../../../../images/icons/icon_3x/chevron-up.png";
import overall_down from "../../../../images/icons/icon_3x/chevron-down.png";


import data_th_1 from "../../../../images/thumbnail/data_th_1.PNG";   // 칫솔
import data_th_2 from "../../../../images/thumbnail/data_th_2.PNG";   // 용접 파이프
import data_th_3 from "../../../../images/thumbnail/data_th_3.PNG";   // 실생활
import data_th_4 from "../../../../images/thumbnail/data_th_4.PNG";   // 폐결핵 판독

import target_th_1 from "../../../../images/thumbnail/RK3399Pro.jpeg";      // RK 3399
import target_th_2 from "../../../../images/thumbnail/JetsonNano.jpeg";     // jetsonnano
import target_th_3 from "../../../../images/thumbnail/odroid-m1B.jpg";      // Odroid
import target_th_4 from "../../../../images/thumbnail/x86PCwitnGPU.jpeg";   // x86
import target_th_5 from "../../../../images/thumbnail/GCP.jpeg";            // GCP
import target_th_6 from "../../../../images/thumbnail/아마존.jpeg";          // AWS

import autonn_mp4 from "../../../../videos/autonn/AutoNN_v3.mp4";                  // autonn.mp4 파일
import labeling_mp4 from "../../../../videos/labeling/LabelingTool.mp4";          // LabelingTool.mp4 파일



function NeuralAndLoadPage({project_id, project_name, project_description})
{
    const [proj_id, setProj_id] = useState(project_id);                                   // 프로젝트 ID
    const [proj_description, setProj_description] = useState(project_description);                 // 프로젝트 설명

    const [dataSet, setDataSet] = useState('');                                 // 데이터 셋 경로
    const [dataset_list, setDataset_list] = useState([]);                       // 데이터 셋 리스트

    const [target, setTarget] = useState('');                                   // 타겟 변경
    const [target_list, setTarget_list] = useState([]);                         // 타겟 리스트

    const [step, setStep] = useState(0);                                        // 신경망 생성 단계
    const [stepText, setStepText] = useState('');                               // 신경망 생성 단계 상황

    const [panel, setPanel] = useState('block');                                // panel 창 상태

    const [dataSetVideoRun, setDataSetVideoRun] = useState(false);

    const target_name = {
        "rk3399pro": "온디바이스 ( RK3399Pro )",
        "jetsonnano": "온디바이스 ( Jetson Nano )",
        "odroid": "온디바이스 ( Odroid-M1 )",
        "x86-cuda": "엣지클라우드 ( x86_64 / CUDA )",
        "gcp": "클라우드 ( GCP )",
        "aws": "클라우드 ( AWS )",
    }


    useEffect( () => {
        console.log('useEffect');

        Request.requestProjectInfo(project_id).then(result =>
        {
            // DB 정보 수신
            if (result.data['target'] !== null && target === '')
            {
                setTarget(result.data['target'])                        // 선택 타겟 정보
            }

            if(result.data['dataset_path'] !== '' && dataSet === '')
            {
                setDataSet(result.data['dataset_path'])                              // 데이터 셋 정보
            }
            setDataset_list(result.data['dataset_list'])

            // 타겟 리스트 이름 변경
            var target_change_name_list = []
            for (var i=0; i < result.data['target_list'].length; i++)
            {
                target_change_name_list.push(target_name[result.data['target_list'][i]]);
            }
            setTarget_list(target_change_name_list);

            // 신경망 생성 단계 프로그래스 바 업데이트
            if(result.data['step'] !== 0)
            {
                setStep(result.data['step'])                            // 신경망 생성 단계

                step_progress_bar_update(result.data['step'], false)

                setPanel('none')

                switch(result.data['step'])
                {
                    case 1:
                        setStepText('')
                        break
                    case 2:
                        setStepText('신경망 생성 완료 --- 실행 이미지 생성 가능')
                        break;
                    case 3:
                        setStepText('실행 이미지 생성 완료 --- 실행 이미지 다운로드 가능')
                        break;
                    case 4:
                        setStepText('실행 이미지 다운로드 완료 --- 타겟 원격 실행 가능')
                        break;
                    case 5:
                        setStepText('타겟 원격 실행 완료')
                        break;
                    default:
                        setStepText('')
                        break;
                }
            }
        })
        .catch(error =>
        {
            console.log('project info get error')
        });
    }, []);


    const save_button_click = () => {
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

     // 데이터 셋 더블 클릭 이벤트
     const dataSetDoubleClick = () =>
     {
        setDataSetVideoRun(!dataSetVideoRun)

        if(panel === 'block' && !dataSetVideoRun === true)
        {
            setPanel('none')
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_down + "')";
        }
     }

     // 데이터 셋 비디오 실행 완료
     const dataSetVideoEnd = () =>
     {
        setDataSetVideoRun(false)

        if(panel === 'none')
        {
            setPanel('block')
            document.getElementById('overall_icon').style.backgroundImage = "url('" + overall_up + "')";
        }
     }

     // 신경망 생성 비디오 재생 완료
     const autonnVideoEnd = () =>
     {
        console.log('autonnVideoEnd');
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
     const targetClick = (value, index) =>
     {
        const target_key = Object.keys(target_name).find(key => target_name[key] === value);
        setTarget(target_key)                // 사용자 선택 타겟 정보

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

    // 타겟 이미지 가져오기
    const getTarget_image = (value) =>
    {
        if (value === "클라우드 ( GCP )" || value === "gcp")
        {
            return target_th_5;
        }
        else if (value === "클라우드 ( AWS )" || value === "aws")
        {
            return target_th_6;
        }
        else if (value === "엣지클라우드 ( x86_64 / CUDA )" || value === "x86-cuda")
        {
            return target_th_4;
        }
        else if (value === "온디바이스 ( Odroid-M1 )" || value === "odroid")
        {
            return target_th_3;
        }
        else if (value === "온디바이스 ( Jetson Nano )" || value === "jetsonnano")
        {
            return target_th_2;
        }
        else if (value === "온디바이스 ( RK3399Pro )" || value === "rk3399pro")
        {
            return target_th_1;
        }
        else
        {
            return "";
        }
    }

    // 'Run' 버튼 클릭 이벤트
    const runButtonClick = () =>
    {
        setDataSetVideoRun(false);

        if(step === 0)
        {
            setStepText('')
            createNeuralNetwork();

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
                createNeuralNetwork();

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

    // 신경망 프로세스 초기화
//    const project_init = () =>
//    {
//        createNeuralNetwork();
////        databaseUpdate(1);
//    }

    // 단계 프로그래스바 업데이트
    const step_progress_bar_update = (current_step, b_step_status) =>
    {
        var progress_bar = document.getElementById('progressbar');

        for (var i = 1; i < progress_bar.childNodes.length + 1; i++)
        {
            if( i < current_step )
            {
                document.getElementById('progress_' + i).className = 'stepper-item2 completed';
            }
            else if( i === current_step )
            {
                if(b_step_status === false)
                {
                    document.getElementById('progress_' + i).className = 'stepper-item2 ready';
                }
                else
                {
                    document.getElementById('progress_' + i).className = 'stepper-item2 active';

                }
            }
            else
            {
                document.getElementById('progress_' + i).className = 'stepper-item2 before';
            }
        }
    }

    // 데이터베이스 업데이트
    const databaseUpdate = (num) =>
    {
        const param = {
            'project_id' : project_id,
            'project_name' : project_name,
            'selectTarget' : target,
            'dataset_path' : dataSet,
            'step' : num
        }

        // 데이터베이스 업데이트
        Request.requestProjectUpdate(param).then(result =>
        {
            setStep(num)
        })
        .catch(error =>
        {
            console.log(error);
        });

        if (num === 1)
        {
            step_progress_bar_update(num, true)
        }
        else
        {
            step_progress_bar_update(num, false)
        }

        if (num === 2)
        {
            setStepText('신경망 생성 완료 --- 실행 이미지 생성 가능')
        }
        else if ( num === 3)
        {
            setStepText('실행 이미지 생성 완료 --- 실행 이미지 다운로드 가능')
        }
        else if ( num === 4)
        {
            setStepText('실행 이미지 다운로드 완료 --- 타겟 원격 실행 가능')
        }
        else if ( num === 5)
        {
            setStepText('타겟 원격 실행 완료')
        }
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
                default:
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
        setStepText('신경망 생성중')
        step_progress_bar_update(1, true)

        setTimeout(function()
        {
            databaseUpdate(2)
        }, 119000);
    }

    // 실행 이미지 생성
    const createRunImageFile = () =>
    {
        setStepText('실행 이미지 생성중')
        step_progress_bar_update(2, true)

        const param = {
            'source_image_path': '소스 이미지 경로',
            'target_image_save_path': '타겟 이미지 저장 경로',
            'target_name': target,
        }

        setTimeout(function()
        {
            // 신경망 생성 완료
            databaseUpdate(3)
        }, 5000);
    }

    // 실행 이미지 탑재
    const deployRunImage = () =>
    {
        setStepText('실행 이미지 다운로드중')

        step_progress_bar_update(3, true)

        const param = {
            'target_image_save_path': '',
            'target_url': '타겟 URL',
            'startup_command' : './rk3399pro-onnx-detector --model=best_detnn-coco-rk3399pro.onnx'
        }

        setTimeout(function()
        {
            // 신경망 생성 완료
            databaseUpdate(4)
        }, 5000);
    }

    // 신경망 실행
    const runNeuralNetwork = () =>
    {
       setStepText('타겟 원격 실행중')

        step_progress_bar_update(4, true)

        setTimeout(function()
        {
            // 신경망 생성 완료
            databaseUpdate(5)
        }, 5000);
    }


    return (
        <>
        {/* 프로젝트 생성 - 신경망 생성 폼 */}
        <div className='project_manage_container'>

            <div className='project_manage_content' >
                <div id="accordion" className="accordion" onClick={ ()=> accordionButtonClick() } style={{height:'40px', backgroundColor:'#303030', borderRadius:panel === 'none'? '5px 5px 5px 5px' : '5px 5px 0px 0px', lineHeight:'0', display:'flex'}}>
                    <span style={{fontSize:'16px', color:'white'}}>Information </span>
                    <div id="overall_icon" className="overall_icon" style={{backgroundImage:panel === 'none'? "url('" + overall_down + "')" : "url('" + overall_down + "')"}}></div>
                </div>

                <div className="panel" style={{display:panel}}>

                    <div className="project_description" style={{backgroundColor:'#303030'}}>
                        <div className="description-content" style={{ padding:'10px 20px 10px 20px', height:'100%', backgroundColor:'#303030', display:'flex'}}>
                            <span style={{color:'white'}}>Description</span>
                            <input onChange={({ target: { value } }) => setProj_description(value)} value={proj_description} style={{ height:'100%', width:'100%', borderRadius:'5px', marginLeft:'20px', marginRight:'20px', fontSize:'16px'}} />
                            <button onClick={() => save_button_click()} style={{ height:'100%', width:'150px', borderRadius:'5px', backgroundColor:'#4A80FF', color:'white', fontSize:'16px', border:'0px'}}>저장</button>
                        </div>
                    </div>

                    <div className="project_user_requirement" style={{display:'flex'}}>

                        {/* 데이터셋 선택 */}
                        <div className='project_user_requirement_left' style={{ padding:'0px 0px 0px 0px', marginRight:'0px'}}>

                            <div className='select_dataset' style={{padding:'0px 0px 0px 20px', color:'black', display:'flex', alignItems:'center', height:'50px'}}>
                                <div style={{fontSize:'16px', width:'auto'}}>Dataset</div>
                            </div>

                            <div className='dataset_content' style={{display:'block', overflow:'auto', paddingBottom:'60px'}}>

                                { dataset_list.length > 0 ?
                                    <>
                                    <div className='dataset_list' style={{height:'100%', width:'100%', padding:'0px 20px 20px 20px', backgroundColor:'white'}}>
                                        {dataset_list.map((menu, index) => {
                                            return (
                                             <div key={index} className={dataSet === menu ? "dataset_item_box tooltip select" : "dataset_item_box tooltip"} onClick={ ()=> dataSetClick(menu, index)} onDoubleClick={ ()=> dataSetDoubleClick()}>
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
                           <div className='select_target' style={{padding:'0px 0px 0px 20px', color:'black', display:'flex', alignItems:'center', height:'20%'}}>
                                <div style={{ fontSize:'16px',  width:'auto'}}>Target</div>
                            </div>

                            <div className='dataset_content' style={{display:'block', overflow:'auto', paddingBottom:'60px'}}>

                                { target_list.length > 0 ?
                                    <>
                                    <div className='dataset_list' style={{height:'100%', width:'100%', padding:'0px 20px 20px 20px', backgroundColor:'white'}}>
                                        {target_list.map((menu, index) => {
                                            return (
                                              <div className={target === menu ? "target_item_box tooltip select" : "target_item_box tooltip"} key={index} onClick={ ()=> targetClick(menu, index)}>
                                                <img id="dataset_item_image" className="dataset_item_image" src={getTarget_image(menu)} style={{height:'100%', width:'100%', margin:'auto', marginRight:'5px', backgroundColor:'#DEDEDE'}}/>
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
                                        Does Not Exist Targets!
                                    </div>
                                    </>
                                }
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



                <div id='project_bottom' className='project_bottom'  style={{padding:'20px 0px 0px 0px', height:'100%',marginBottom:'0px', gridRow: panel === 'none' ? '2/4' : '3/4' }}>

                    <div className='create_neural_network' style={{ backgroundColor:'#303030', borderRadius:'5px', height:'100%', padding:'10px 20px 20px 20px'}}>
                        <div style={{marginBottom:'10px', display:'flex'}}>
                            <span style={{fontSize:'16px', color:'white'}}>Create Step - </span>
                            <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}>[ </span>
                            <span style={{color:'#4A80FF'}}>{stepText}</span>{stepText.indexOf('중') !== -1 && <div className="loader" style={{marginLeft:'10px'}}></div>}
                            <span style={{color:'white', marginLeft:'10px', marginRight:'10px'}}> ]</span>

                        </div>
                        <div className='status_level' style={{backgroundColor:'white', padding:'15px 0px', borderRadius:'5px'}}>
                            <div className="stepper-wrapper2" id='progressbar'>
                                <div className="stepper-item2 before" id='progress_1'>
                                    <div className="step-counter2" onClick={() => progress_button_click(1)}>신경망 자동 생성</div>
                                </div>
                                <div className="stepper-item2 before" id='progress_2'>
                                    <div className="step-counter2" onClick={() => progress_button_click(2)}>실행 이미지 생성</div>
                                </div>
                                <div className="stepper-item2 before" id='progress_3'>
                                    <div className="step-counter2" onClick={() => progress_button_click(3)}>실행 이미지 다운로드</div>
                                </div>
                                <div className="stepper-item2 before" id='progress_4'>
                                    <div className="step-counter2" onClick={() => progress_button_click(4)}>타겟 원격 실행</div>
                                </div>
                            </div>
                        </div>

                        <div className='status_log' style={{color:'white', height:'auto', overflow:'auto',  padding:'20px 0px 0px 0px'}}>

                            <div style={{ border:'2px solid white', borderRadius:'5px', backgroundColor:'white', position:'relative', width:'100%', height:'100%'}}>
                            {/*
                                {stepText.indexOf('신경망 생성중') !== -1 &&
                                     <video autoPlay={true} id='status_result'  style={{borderRadius:'5px', position:'absolute', top:'0', left:'0', width:'100%', height:'100%', objectFit:'fill'}}>
                                        <source src={autonn_mp4} type="video/mp4"/>
                                     </video>
                                 }
                            */}

                                {stepText.indexOf('신경망 생성중') !== -1 &&
                                    <div style={{width:'100%', height:'100%', display:stepText.indexOf('신경망 생성중') !== -1 ? 'block' : 'none'}}>
                                         <div style={{backgroundColor:'white', width:'15%', height:'100%', float:'left', padding:'5px 10px 10px 10px'}}>
                                            <div style={{borderRadius:'5px', backgroundColor:'#303030', width:'100%', height:'50%', padding:'10px', marginBottom:'5px'}}>
                                                <div className='select_dataset' style={{padding:'0px 0px 0px 0px', color:'white', alignItems:'center', height:'10%'}}>
                                                    <div style={{ fontSize:'16px',  width:'auto'}}>Select Dataset</div>
                                                </div>

                                                <div className="select_data_image tooltip" style={{ borderRadius:'5px',  height:'90%', width:'100%', backgroundColor:'#DEDEDE', marginBottom:'10px'}}>
                                                    <div id="data_image" className="data_image" style={{height:'100%', width:'100%', backgroundImage:dataSet !== '' && "url('" + getDataset_image(dataSet) + "')" }}/>
                                                    <span className="select_tooltiptext" style={{width:'200px'}}>{dataSet}</span>
                                                </div>
                                            </div>

                                            <div style={{borderRadius:'5px', backgroundColor:'#303030', width:'100%', height:'50%', padding:'10px'}}>
                                                <div className='select_target' style={{padding:'0px 0px 0px 0px', color:'white', alignItems:'center', height:'10%'}}>
                                                    <div style={{ fontSize:'16px',  width:'auto'}}>Select Target</div>
                                                </div>

                                                <div className="select_target_image tooltip" style={{ borderRadius:'5px',  height:'90%', width:'100%', backgroundColor:'#DEDEDE'}}>
                                                    <div id="target_image" className="target_image" style={{height:'100%', width:'100%', margin:'auto', backgroundImage:target !== '' && "url('" + getTarget_image(target) + "')" }}/>
                                                    <span className="select_tooltiptext" style={{width:'220px'}}>{target_name[target]}</span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className='videoArea' style={{border:'3px solid #303030', width:'85%', height:'100%', float:'right'}}>
                                             <video autoPlay={true} id='status_result' onEnded={() => autonnVideoEnd()} style={{top:'0', left:'0', width:'100%', height:'100%', objectFit:'fill'}}>
                                                <source src={autonn_mp4} type="video/mp4"/>
                                             </video>
                                         </div>
                                    </div>
                                }

                                 {dataSetVideoRun === true &&
                                     <video autoPlay={true} id='status_result' onEnded={() => dataSetVideoEnd()} style={{position:'absolute', top:'0', left:'0', width:'100%', height:'100%', objectFit:'fill'}}>
                                        <source src={labeling_mp4} type="video/mp4" />
                                     </video>
                                 }
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


