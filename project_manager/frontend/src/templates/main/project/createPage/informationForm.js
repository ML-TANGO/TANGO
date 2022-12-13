import React from "react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/project_management.css'

import imageCompression from "browser-image-compression";

import * as RequestTarget from "../../../../service/restTargetApi";
import * as RequestLabelling from "../../../../service/restLabellingApi";

import * as TargetPopup from "../../../../components/popup/targetPopup";

function InformationForm(
    {dataset, setDataset,
    dataset_list, setDataset_list,
    target, setTarget,
    target_list, setTarget_list,
    setTaskType})
{
//    const [dataset_list, setDataset_list] = useState([]);              // 데이터 셋 리스트
//    const [target_list, setTarget_list] = useState([]);                // 타겟 리스트

    /* 타겟 생성 정보 */
    const [target_name, setTarget_name] = useState('');                               // 타겟 이름
    const [target_image, setTarget_image] = useState('');                             // 타겟 이미지
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

    useEffect( () =>
    {
        get_dataset_list()
        get_target_list()
    }, []);

    // 데이터 셋 정보 수신 - 레이블링 저작도구 연동data['target']
    const get_dataset_list = () =>
    {
        RequestLabelling.requestDatasetList().then(result =>
        {
            let resKeys = Object.keys(result.data)

            let resList = []
            for ( let d in resKeys)
            {
                let info = result.data[d]

                // 생성 완료된 데이터 셋만 추가
                if(info.DATASET_STS === 'DONE' )
                {
                    resList.push(info)
                }
            }
            setDataset_list(resList)
        })
        .catch(error =>
        {
            console.log('Data set list get error')
        });
    }

    // 데이터 셋 선택
    const dataSetClick = (value, index) =>
    {
        // 선택한 데이터 셋의 타입 정보에 따라 Task Type 정보 설정
        switch(value.OBJECT_TYPE)
        {
            case "C":
                setTaskType('classification');
                break;
            case "D":
                setTaskType('detection');
                break;
        }

        setDataset(value);
    }

    // 데이터 셋의 Type 이름 변경
    const getTypeName = (type) =>
    {
        let type_name = ''
        switch(type)
        {
            case "C":
                type_name = 'classification'
                break;
            case "D":
                type_name = 'detection'
                break;
            case "I":
                type_name = 'image'
                break;
            case "V":
                type_name = 'video'
                break;
            default:
                break
        }
        return type_name
    }

    // 데이터 셋 이미지 가져오기
    const getDataset_image = (value) =>
    {
          const host = window.location.hostname
          const imageAddress = 'http://' + host + ':8095' + value

          return imageAddress
    };

    // 타겟 리스트 수신
    const get_target_list = () =>
    {
        RequestTarget.requestTargetList().then(result =>
        {
            setTarget_list(result.data)
        })
        .catch(error =>
        {
            console.log('target list get error')
        });
    }

    // 타겟 선택
    const targetClick = (selectTarget, index) =>
    {
        setTarget(selectTarget)
    }

    // 타겟 추가 버튼 클릭
    const target_add_button_click = () =>
    {
       document.getElementById('create_target_popup').style.display = 'block';
    }

    // 타겟 생성 입력 폼 정보 초기화
    const stateInfoInit = () =>
    {
        setTarget_name('')
        setTarget_image('')
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

    // 타겟 이미지 업로드
    const uploadImageFile = async (event) =>
    {
        const file = event.target.files[0];

        const options = {
                maxSizeMB: 0.2,
                maxWidthOrHeight: 720,
                useWebWorker: true
        }

        const compressFile = await imageCompression(file, options);

        const reader = new FileReader();

        reader.onload = function() {
            setTarget_image(reader.result);
            setPreviewURL(reader.result)
        }

        reader.readAsDataURL(compressFile);

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
            'name': target_name,
            'image': target_image,
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
            console.log('target create error')
        });
    }


    return (
        <>
        {/* 타겟 생성 팝업 */}
        <TargetPopup.TargetCreatePopup
            popup_modify_mode={false}
            target_name={target_name}
            setTarget_name={setTarget_name}
            previewURL={previewURL}
            target_image={target_image}

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
                                    <div key={index} className={dataset.DATASET_CD === menu.DATASET_CD ? "dataset_item_box tooltip select" : "dataset_item_box tooltip"} onClick={ ()=> dataSetClick(menu, index)}>
                                        <img id="dataset_item_image" className="dataset_item_image" src={getDataset_image(menu.THUM_NAIL)} style={{height:'100%', width:'100%', margin:'auto', backgroundColor:'#DEDEDE'}}/>

                                        {/* <span className="dataset_tooltiptext" style={{width:'150px'}}>{menu.TITLE}</span> */}

                                        <span className="dataset_tooltip"></span>

                                        <div className="dataset_tooltip_text">
                                             <div className="dataset_inner_text">- Title : {menu.TITLE}</div>
                                             <div className="dataset_inner_text">- Format : {getTypeName(menu.DATA_TYPE)}</div>
                                             <div className="dataset_inner_text">- Type : {getTypeName(menu.OBJECT_TYPE)}</div>
                                             <div className="dataset_inner_text">- Files : {menu.FILE_COUNT}</div>
                                        </div>


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
                        <div className='target_list' style={{height:'100%', width:'100%', padding:'0px 20px 20px 20px', backgroundColor:'white',  maxHeight:'160px', overflow:'auto'}}>
                            {target_list.map((menu, index) => {
                                return (
                                    <div className={menu.id === target.id ? "target_item_box tooltip select" : "target_item_box tooltip"} key={index} onClick={ ()=> targetClick(menu, index)}>
                                        <img id="target_item_image" className="target_item_image" src={menu.image} style={{height:'100%', width:'100%', margin:'auto', backgroundColor:'#DEDEDE'}}/>
                                        {/* <span className="dataset_tooltiptext" style={{width:'150px'}}>{menu.name}</span> */}

                                        <span className="target_tooltip"></span>

                                        <div className="target_tooltip_text">
                                             <div className="target_inner_text">- Name : {menu.name}</div>
                                             <div className="target_inner_text">- Info : {menu.info}</div>
                                             <div className="target_inner_text">- Engine : {menu.engine}</div>
                                             <div className="target_inner_text">- OS : {menu.os}</div>
                                             <div className="target_inner_text">- CPU : {menu.cpu}</div>
                                        </div>

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
        </>
    );
}

export default InformationForm;