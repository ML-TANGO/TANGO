import React from "react";
import { useEffect, useState } from "react";

import { PlusLg } from "react-bootstrap-icons";

import Kebab from "../../../components/Kebab/Kebab";
import * as TargetPopup from "../../../components/popup/targetPopup";

import '../../../CSS/project_management.css'
import '../../../CSS/target_management.css'

import * as Request from "../../../service/restTargetApi";

function TargetMain()
{
    const [popup_modify_mode, setPopup_modify_mode] = useState(false);

    const [targetList, setTargetList] = useState([]);

    const [target_image_list, setTarget_image_list] = useState([]);

    const [selectTargetId, setSelectTargetId] = useState(0);            // 수정 타겟 ID

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

    /* 페이지 로드 완료시 호출 이벤트 */
    useEffect( () =>
    {
        getTarget_list();
    }, []);

    const stateInfoInit = () =>
    {
        setSelectTargetId(0)
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

    const getTarget_list = () =>
    {
        Request.requestTargetList().then(result =>
        {
            setTargetList(result.data);
        })
        .catch(error =>
        {
            console.log(error)
            console.log('target list get error')
        });
    };


    // 타겟 생성 버튼 클릭
    const target_createButtonClick = () =>
    {
        setPopup_modify_mode(false);

        document.getElementById('create_target_popup').style.display = 'block';     // 생성 팝업 보이기
    };

    /* 생성 팝업 - 취소 버튼 클릭 */
    const target_popup_Cancel_ButtonClick = () =>
    {
        stateInfoInit();
        /* 팝업 숨김 */
        document.getElementById('create_target_popup').style.display = 'none';
    }

    // 타겟 삭제
    const targetDelete = (id) =>
    {
        Request.requestTargetDelete(id).then(result =>
        {
            getTarget_list();
        })
        .catch(error =>
        {
            console.log('target delete error')
        });
    };


    // 타겟 수정
    const targetModify = (selectTarget) => {
        console.log('targetModify');

        setPopup_modify_mode(true);

        setSelectTargetId(selectTarget.id);
        setTargetName(selectTarget.name);
        setTargetImage(selectTarget.image);
        setPreviewURL(selectTarget.image);

        setTarget_info(selectTarget.info)
        setTarget_engine(selectTarget.engine)
        setTarget_os(selectTarget.os)
        setTarget_cpu(selectTarget.cpu)
        setTarget_acc(selectTarget.acc)
        setTarget_memory(selectTarget.memory)
        setTarget_host_ip(selectTarget.host_ip)
        setTarget_host_port(selectTarget.host_port)
        setTarget_host_service_port(selectTarget.host_service_port)

        document.getElementById('create_target_popup').style.display = 'block';
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

        Request.requestTargetCreate(param).then(result =>
        {
            document.getElementById('create_target_popup').style.display = 'none';      // 생성 팝업 숨기기

            stateInfoInit();

            getTarget_list();
        })
        .catch(error =>
        {
            console.log('description modify error')
        });
    }


    // 타겟 수정 팝업 확인 버튼 클릭
    const modify_popup_Apply_ButtonClick = () => {

        const result = window.confirm("타겟을 수정하시겠습니까?");

        if(result)
        {
            const param = {
                'id': selectTargetId,
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

            Request.requestTargetUpdate(param).then(result =>
            {
                stateInfoInit();

                getTarget_list();

                document.getElementById('create_target_popup').style.display = 'none';      // 생성 팝업 숨기기
            })
            .catch(error =>
            {
                console.log('target update error')
            });
        }
    };

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

    return (
        <>
        {/* 타겟 생성 팝업 */}
        <TargetPopup.TargetCreatePopup
            popup_modify_mode={popup_modify_mode}
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
            create_ButtonClick={target_popup_Create_ButtonClick}
            apply_ButtonClick={modify_popup_Apply_ButtonClick}/>

        <div className='manage_list_container_sub'>
            {/* 타겟 메인 페이지 */}
            <div className='manage_container'>

                {/* 타겟 메인 페이지 - 헤더 */}
                <div className='manage_header' style={{width:'100%'}}>
                    {/* 이동경로 레이아웃 */}
                    <div className='path'></div>

                    <div className='title'>
                        <div className='title_left'>Target Management</div>

                        <div className='title_right'>
                            <div onClick={ () => target_createButtonClick() }>
                                <button type='button' id='create_button'> <PlusLg size={19} color="#C3c8d3" />&nbsp; Create Target</button>
                            </div>
                        </div>
                    </div>

                </div>

                {/* 타겟 메인 페이지 - 바디 */}
                <div className='target_manage_content' style={{height:'100%', borderRadius:'5px'}}>

                    <div className='target_list' style={{backgroundColor:'#303030', borderRadius:'5px', height:'100%'}}>

                        <div className="target_list_title" style={{ padding:'10px 20px 10px 20px', height:'100%',  borderRadius:'5px', backgroundColor:'#303030'}}>
                            <span style={{color:'white'}}>Target List</span>
                        </div>

                        <div className="target_list_body" style={{ padding:'0px 20px 20px 20px', height:'100%', backgroundColor:'#303030'}}>
                            <div style={{ height:'100%', borderRadius:'5px', backgroundColor:'white', overflow:'auto'}}>

                            { targetList.length > 0 ?
                                <>
                                <div className="target-content" style={{padding:'20px 20px 20px 20px', overflow:'auto'}}>
                                    {targetList.map((target, index) => {
                                        return (
                                            <div className="target-item" key={index} style={{backgroundColor:'#303030'}}>
                                            {/*<div className="target-item" key={index} style={{backgroundColor:'#303030'}} onClick={() => target_item_click()}>*/}
                                                <div className="target-item-image">

                                                    <img className="target-image" src={target.image} style={{width:'110px', height:'110px'}}/>
                                                </div>

                                                <div className="target-item-content">
                                                    <Kebab index={index} page={'target'} itemID={target.id} deleteItem={targetDelete} modifyItem={() => targetModify(target)} deleteAlter={target.name + " 타겟을"}/>
                                                    <div id="" style={{ color: "white", fontSize: "24px", height:"40px", borderBottom:"1px solid"}}>{target.name}</div>


                                                    <div className="target-info" style={{color: "#c1c3c7", width:"100%", height:"calc(100% - 40px)"}}>
                                                        <div className="target-cpu" style={{ textAlign:'center', borderRight:'1px solid'}}>
                                                            <div style={{fontSize: "18px", height:'50%'}}>Info</div>
                                                            <div style={{fontSize: "14px", height:'50%', overflow:'auto'}}>{target.info}</div>
                                                        </div>
                                                        <div className="target-gpu" style={{textAlign:'center', borderRight:'1px solid'}}>
                                                            <div style={{fontSize: "18px", height:'50%'}}>Engine</div>
                                                            <div style={{fontSize: "14px", height:'50%', overflow:'auto'}}>{target.engine}</div>
                                                        </div>
                                                        <div className="target-memory" style={{textAlign:'center', borderRight:'1px solid'}}>
                                                            <div style={{fontSize: "18px", height:'50%'}}>Memory</div>
                                                            <div style={{fontSize: "14px", height:'50%', overflow:'auto'}}>{target.memory}</div>
                                                        </div>
                                                        <div className="target-model" style={{textAlign:'center'}}>
                                                            <div style={{fontSize: "18px", height:'50%'}}>OS</div>
                                                            <div style={{fontSize: "14px", height:'50%', overflow:'auto'}}>{target.os}</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                                </>
                                :
                                <>
                                {/* 프로젝트 리스트가 없는 경우 */}
                                <div style={{height:'100%', width:'100%',
                                    fontSize:"50px", fontWeight:'700', textAlign:'center',
                                    display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                                    Please Create Target!
                                </div>
                                </>
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

export default TargetMain;