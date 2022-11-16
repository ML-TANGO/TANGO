import React from "react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/project_management.css'

import imageCompression from "browser-image-compression";

//import data_th_1 from "../../../../images/thumbnail/data_th_1.PNG";         // 칫솔
//import data_th_2 from "../../../../images/thumbnail/data_th_2.PNG";         // 용접 파이프
//import data_th_3 from "../../../../images/thumbnail/data_th_3.PNG";         // 실생활
//import data_th_4 from "../../../../images/thumbnail/data_th_4.PNG";         // 폐결핵 판독

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

    // 현재 페이지 정보가 변경될 경우 반복 호출
    useEffect(() =>
    {
        get_dataset_list()
        get_target_list()
    }, []);

    // 데이터 셋 정보 수신 - 레이블링 저작도구 연동
    const get_dataset_list = () =>
    {
        RequestLabelling.requestDatasetList().then(result =>
        {
            let resKeys = Object.keys(result.data)

            let resList = []
            for ( let d in resKeys)
            {
                let info = result.data[d]

                // 삭제된 데이터 셋이 아닌 경우
                if(info.DATASET_STS !== 'DELETE')
                {
                    resList.push(info)
                }
            }
            setDataset_list(resList)

            const data_index = resList.findIndex(d => d === dataset)
            if(data_index !== -1) setDataset(resList[data_index])

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

        setDataset(value.DATASET_CD);
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
          const host = window.location.hostname
          const imageAddress = 'http://' + host + ':8095' + value

          return imageAddress
    };

    // 타겟 리스트 수신
    const get_target_list = () =>
    {
        RequestTarget.requestTargetList().then(result =>
        {
            setTarget_list(result.data);

            const target_index = result.data.findIndex(t => t.id === target)
            if(target_index !== -1) setTarget(parseInt(result.data[target_index]))
        })
        .catch(error =>
        {
            console.log('target list get error')
        });
    }

    // 타겟 선택
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

    // 타겟 추가 버튼 클릭
    const target_add_button_click = () =>
    {
       document.getElementById('create_target_popup').style.display = 'block';
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

    // 타겟 이미지 업로드
    const uploadImageFile = async (event) =>
    {
        const file = event.target.files[0];

        const options = {
                maxSizeMB: 0.2,
                maxWidthOrHeight: 1920,
                useWebWorker: true
        }

        const compressFile = await imageCompression(file, options);

        const reader = new FileReader();

        reader.onload = function() {
            setTargetImage(reader.result);
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
                                 <div key={index} className={dataset === menu.DATASET_CD ? "dataset_item_box tooltip select" : "dataset_item_box tooltip"} onClick={ ()=> dataSetClick(menu, index)}>
                                    <img id="dataset_item_image" className="dataset_item_image" src={getDataset_image(menu.THUM_NAIL)} style={{height:'100%', width:'100%', margin:'auto', marginRight:'5px', backgroundColor:'#DEDEDE'}}/>
                                    <span className="dataset_tooltiptext" style={{width:'150px'}}>{menu.TITLE}</span>
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
        </>
    );
}

export default InformationForm;