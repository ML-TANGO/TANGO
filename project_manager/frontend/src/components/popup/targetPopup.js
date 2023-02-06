import React from "react";
import { useEffect } from "react";

import Select from 'react-select';

import '../../CSS/popup.css'

/* 프로젝트 생성 팝업 */
export function TargetCreatePopup({
    popup_modify_mode,
    targetName, setTargetName,
    targetImage, previewURL,
    target_info, setTarget_info,
    target_engine, setTarget_engine,
    target_os, setTarget_os,
    target_cpu, setTarget_cpu,
    target_acc, setTarget_acc,
    target_memory, setTarget_memory,
    target_host_ip, setTarget_host_ip,
    target_host_port, setTarget_host_port,
    target_host_service_port, setTarget_host_service_port,
    uploadImageFile,
    cancel_ButtonClick,
    create_ButtonClick,
    apply_ButtonClick})
{

    const get_os = [
        { value: 'windows', label: 'windows' },
        { value: 'ubuntu', label: 'ubuntu' },
        { value: 'android', label: 'android' }
    ]

    const get_cpu = [
        { value: 'arm', label: 'ARM' },
        { value: 'x86', label: 'X86' }
    ]

    const get_acc = [
        { value: 'cuda', label: 'cuda' },
        { value: 'opencl', label: 'opencl' },
        { value: 'cpu', label: 'cpu' }
    ]

    useEffect( () =>
    {
        if(target_os !== '')
        {
            // os 설정
            get_os.forEach(v => {
                 if(v.value === target_os)
                 {
                    setTarget_os(v);
                 }
            });
        }

        if(target_cpu !== '')
        {
             // cpu 설정
            get_cpu.forEach(v => {
                 if(v.value === target_cpu)
                 {
                    setTarget_cpu(v);
                 }
            });
        }

        if(target_acc !== '')
        {
            // acc 설정
            get_acc.forEach(v => {
                 if(v.value === target_acc)
                 {
                    setTarget_acc(v);
                 }
            });
        }

        const infoRadioList = document.getElementsByName('target_info_radio');

        for (let i = 0; i < infoRadioList.length; i++)
        {
            if(infoRadioList[i].value === target_info)
            {
                infoRadioList[i].checked = true;
                break;
            }
        }

    });


    const changeTargetInfo = (value) => {

        setTarget_info(value);

        if (value === 'pc' || value === 'cloud')
        {
            setTarget_engine('pytorch');
        }
        else{
            setTarget_engine('');
        }
    }


    return (
        <div className='popup' id='create_target_popup' style={{zIndex:'1000'}}>

            <div className="popup_inner">
                {popup_modify_mode === false ?
                    <div className="popup_header" style={{textAlign:'left', fontSize:'16px', color:'#6c7890', fontWeight:'bold'}}>Create Target</div>
                    :
                    <div className="popup_header" style={{textAlign:'left', fontSize:'16px', color:'#6c7890', fontWeight:'bold'}}>Modify Target</div>
                }

                <div className="popup_body">

                    <label className="popup_label" style={{textAlign:'left', margin:'0px'}}>Name</label>
                    <input
                        className="target_input"
                        id='input_target_name'
                        value={targetName}
                        onChange={({ target: { value } }) => setTargetName(value)}
                        type="text"
                        placeholder="Target Name"
                        style={{padding:'0px 0px 0px 10px', marginTop:'5px', marginBottom:'10px'}}
                        maxLength='30'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginBottom:'10px'}}>Image</label>

                    <div style={{width:'100%', height:'150px', backgroundColor:'#303030', marginBottom:'0px', textAlign:'center'}}>
                        {previewURL !== '' &&
                            <img src={previewURL} alt='preview' style={{width:'240px', height:'150px', textAlign:'center'}}/>
                         }
                    </div>

                    <div className="target_filebox" style={{width:'100%', textAlign:'right'}}>
                        <label htmlFor="input_target_image">파일 찾기</label>
                        <input
                            className="target_input"
                            id='input_target_image'
                            onChange={uploadImageFile}
                            type="file"
                            accept='image/jpg,image/png,image/jpeg'
                        />
                    </div>
                    <hr/>

                    <div style={{marginBottom:'20px'}}>
                        <div className="popup_header" style={{textAlign:'left', fontSize:'14px', color:'#6c7890', fontWeight:'bold', marginBottom:'10px'}}>Specification</div>
                        {/* Target Info */}
                        <div style={{display:"flex", width:'100%', height:'40px'}}>
                            <label className="popup_label" style={{textAlign:'left', marginTop:'10px', marginRight:'10px'}}>Target Info : </label>
                            <input type="radio" name="target_info_radio" value="pc" onChange={({ target: { value } }) => changeTargetInfo(value)}/><span style={{marginTop:'10px', marginRight:'10px'}}>PC</span>
                            <input type="radio" name="target_info_radio" value="ondevice" onChange={({ target: { value } }) => changeTargetInfo(value)}/><span style={{marginTop:'10px', marginRight:'10px'}}>OnDevice</span>
                            <input type="radio" name="target_info_radio" value="cloud" onChange={({ target: { value } }) => changeTargetInfo(value)}/><span style={{marginTop:'10px'}}>Cloud</span>
                        </div>

                        {/* Engine */}
                        <div style={{display:"flex", width:'100%', marginTop:'0px', height:'40px'}}>
                            <label className="popup_label" style={{textAlign:'left', marginTop:'10px', marginRight:'10px'}}>Engine : </label>
                            <input type="radio" name="target_engine_radio" value="acl"
                                onChange={({ target: { value } }) => setTarget_engine(value)}
                                disabled={target_info === 'pc' || target_info === 'cloud'}/>
                                <span style={{marginTop:'10px', marginRight:'10px'}}>ACL</span>

                            <input type="radio" name="target_engine_radio" value="rknn"
                                onChange={({ target: { value } }) => setTarget_engine(value)}
                                disabled={target_info === 'pc' || target_info === 'cloud'}/>
                                <span style={{marginTop:'10px', marginRight:'10px'}}>RKNN</span>

                            <input type="radio" name="target_engine_radio" value="pytorch"
                                onChange={({ target: { value } }) => setTarget_engine(value)}
                                checked={target_info === 'pc' || target_info === 'cloud'}
                                disabled={target_info !== 'pc' && target_info !== 'cloud'}/>
                                <span style={{marginTop:'10px', marginRight:'10px'}}>
                                PyTorch
                                </span>
                        </div>
                        <hr/>

                        <div style={{display:'flex', width:'100%'}}>
                            {/* OS */}
                            <div style={{display:"flex", width:'50%', marginTop:'10px', height:'40px'}}>
                                <label className="popup_label" style={{textAlign:'right', marginTop:'10px', marginRight:'10px', width:'30%'}}>OS : </label>
                                <Select options={get_os} isSearchable={false} defaultValue={get_os[0]}
                                value={target_os}
                                onChange={setTarget_os}/>
                            </div>

                            {/* CPU */}
                            <div style={{display:"flex", width:'50%', marginTop:'10px', height:'40px'}}>
                                <label className="popup_label" style={{textAlign:'right', marginTop:'10px', marginRight:'10px', width:'30%'}}>CPU : </label>
                                <Select options={get_cpu} isSearchable={false} defaultValue={get_cpu[0]} style={{width:'100%'}}
                                value={target_cpu}
                                onChange={setTarget_cpu}/>
                            </div>
                        </div>

                        <div style={{display:'flex', width:'100%'}}>
                            {/* ACC */}
                            <div style={{display:"flex", width:'100%', marginTop:'10px', height:'35px'}}>
                                <label className="popup_label" style={{textAlign:'right', marginTop:'10px', marginRight:'10px', width:'30%'}}>Accelerator : </label>
                                <Select options={get_acc} isSearchable={false} defaultValue={get_acc[0]}
                                value={target_acc}
                                onChange={setTarget_acc}/>
                            </div>

                            {/* Memory */}
                            <div style={{display:"flex", width:'100%', marginTop:'10px', height:'40px'}}>
                                <label className="popup_label" style={{textAlign:'right', marginTop:'10px', marginRight:'10px', width:'30%'}}>Memory : </label>
                                <input
                                    className="target_input"
                                    type="number"
                                    placeholder="MB unit"
                                    style={{padding:'0px 0px 0px 10px', marginBottom:'0px', width:'120px'}}
                                    maxLength='100'
                                    value={target_memory}
                                    onChange={({ target: { value } }) => setTarget_memory(value)}/>
                            </div>
                        </div>
                    </div>
                    <hr/>

                    {/* Target Host */}
                    <div style={{marginTop:'20px', marginBottom:'20px', display:target_info === 'pc' || target_info === 'cloud' ? 'block' : 'none'}}>
                        <div className="popup_header" style={{textAlign:'left', fontSize:'14px', color:'#6c7890', fontWeight:'bold', marginBottom:'10px'}}>Target Host</div>

                        {/* Target Host - IP Address */}
                        <div style={{display:"flex", width:'100%', marginTop:'10px', height:'40px'}}>
                            <label className="popup_label" style={{textAlign:'right', marginTop:'10px', marginRight:'10px', width:'20%'}}>IP Address : </label>
                            <input
                                className="target_input"
                                type="text"
                                style={{padding:'0px 0px 0px 10px', marginBottom:'10px', height:'40px'}}
                                placeholder="xxx.yyy.zzz.www"
                                maxLength='100'
                                value={target_host_ip}
                                onChange={({ target: { value } }) => setTarget_host_ip(value)}/>
                        </div>

                        {/* Target Host - Port */}
                        <div style={{display:"flex", width:'100%', marginTop:'10px', height:'40px'}}>
                            <label className="popup_label" style={{textAlign:'right', marginTop:'10px', marginRight:'10px', width:'20%'}}>Port : </label>
                            <input
                                className="target_input"
                                type="text"
                                style={{padding:'0px 0px 0px 10px', marginBottom:'10px', height:'40px'}}
                                maxLength='100'
                                value={target_host_port}
                                onChange={({ target: { value } }) => setTarget_host_port(value)}/>
                        </div>

                        {/* Target Host - Service Port */}
                        <div style={{display:"flex", width:'100%', marginTop:'10px', height:'40px'}}>
                            <label className="popup_label" style={{textAlign:'right', marginTop:'10px', marginRight:'10px', width:'20%'}}>Service Port : </label>
                            <input
                                className="target_input"
                                type="text"
                                style={{padding:'0px 0px 0px 10px', marginBottom:'10px', height:'40px'}}
                                maxLength='100'
                                value={target_host_service_port}
                                onChange={({ target: { value } }) => setTarget_host_service_port(value)}/>
                        </div>
                    </div>

                </div>


                <div className="popup_button_list" style={{width:'100%'}}>
                    <button onClick={ cancel_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#707070'}}>Cancel</button>

                    {/*
                    {popup_modify_mode === false ?
                        <button onClick={ create_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Create</button>
                        :
                        <button onClick={ apply_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Modify</button>
                    }
                    */}

                    {/* 타겟 정보가 cloud가 아닌 경우 */}
                    {target_info !== 'cloud' ?
                        <>
                        {targetName.trim().length > 0 &&
                            previewURL.length > 0 &&
                            target_info !== '' &&
                            target_engine !== '' &&
                            target_os.value !== '' &&
                            target_cpu.value !== '' &&
                            target_acc.value !== '' &&
                            target_memory.trim().length > 0
                        ?
                            <>
                            {popup_modify_mode === false ?
                                <button onClick={ create_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Create</button>
                                :
                                <button onClick={ apply_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Modify</button>
                            }
                            </>
                        :
                            <>
                            {popup_modify_mode === false ?
                                <button style={{width:'95px', height:'35px', float:'right', backgroundColor:'lightgrey', marginRight:'15px'}} disabled>Create</button>
                                :
                                <button style={{width:'95px', height:'35px', float:'right', backgroundColor:'lightgrey', marginRight:'15px'}} disabled>Modify</button>
                            }
                            </>
                        }
                        </>

                    :
                        <>
                        {targetName.trim().length > 0 &&
                            previewURL.length > 0 &&
                            target_info !== '' &&
                            target_engine !== '' &&
                            target_os.value !== '' &&
                            target_cpu.value !== '' &&
                            target_acc.value !== '' &&
                            target_memory.trim().length > 0 &&
                            target_host_ip.trim().length > 0 &&
                            target_host_port.trim().length > 0 &&
                            target_host_service_port.trim().length > 0
                        ?
                            <>
                            {popup_modify_mode === false ?
                                <button onClick={ create_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Create</button>
                                :
                                <button onClick={ apply_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Modify</button>
                            }
                            </>
                        :
                            <>
                            {popup_modify_mode === false ?
                                <button style={{width:'95px', height:'35px', float:'right', backgroundColor:'lightgrey', marginRight:'15px'}} disabled>Create</button>
                                :
                                <button style={{width:'95px', height:'35px', float:'right', backgroundColor:'lightgrey', marginRight:'15px'}} disabled>Modify</button>
                            }
                            </>
                        }
                        </>
                    }

                </div>
            </div>
        </div>
    );
}

