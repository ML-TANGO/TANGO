import React from "react";
import { useEffect, useState } from "react";

import '../../CSS/popup.css'


/* 프로젝트 생성 팝업 */
export function TargetCreatePopup({
    targetName, setTargetName,
    targetImage, previewURL,
    targetSpec_cpu, targetSpec_gpu, targetSpec_memory, targetSpec_model,
    specChange, uploadImageFile,
    cancel_ButtonClick,
    create_ButtonClick})
{
    return (
        <div className='popup' id='create_target_popup'>
            <div className="popup_inner">
                <div className="popup_header" style={{textAlign:'left', fontSize:'16px', color:'#6c7890', fontWeight:'bold'}}>Create Target</div>

                <div className="popup_body">
                    <label className="popup_label" style={{textAlign:'left', margin:'0px'}}>Name</label>
                    <input
                        id='input_target_name'
                        value={targetName}
                        onChange={({ target: { value } }) => setTargetName(value)}
                        type="text"
                        placeholder="Target Name"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'20px'}}
                        maxLength='30'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'0px'}}>Image</label>

                    <div className="target_filebox">

                        <label htmlFor="input_target_image">파일 찾기</label>
                        <input
                            id='input_target_image'
                            /* value={targetImage} */
                            onChange={uploadImageFile}
                            type="file"
                            accept='image/jpg,image/png,image/jpeg'
                            style={{padding:'0px 0px 0px 10px', backgroundColor:'red'}}
                        />
                    </div>

                    <div style={{width:'100%', height:'180px', backgroundColor:'#303030', marginBottom:'30px', textAlign:'center'}}>
                        {previewURL !== '' &&
                            <img src={previewURL} alt='preview' style={{width:'240px', height:'180px', textAlign:'center'}}/>
                         }
                    </div>

                <div className="popup_header" style={{textAlign:'left', fontSize:'14px', color:'#6c7890', fontWeight:'bold', marginBottom:'20px'}}>Specification</div>

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>CPU</label>
                    <input
                        id='input_target_cpu'
                        value={targetSpec_cpu}
                        onChange={({ target: { value } }) => specChange(1, value)}
                        type="text"
                        placeholder="CPU"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>GPU</label>
                    <input
                        id='input_target_gpu'
                        value={targetSpec_gpu}
                        onChange={({ target: { value } }) => specChange(2, value)}
                        type="text"
                        placeholder="GPU"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>Memory</label>
                    <input
                        id='input_target_memory'
                        value={targetSpec_memory}
                        onChange={({ target: { value } }) => specChange(3, value)}
                        type="text"
                        placeholder="Memory"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>Model</label>
                    <input
                        id='input_target_model'
                        value={targetSpec_model}
                        onChange={({ target: { value } }) => specChange(4, value)}
                        type="text"
                        placeholder="Model"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />
                </div>
                <div className="popup_button_list" style={{width:'100%'}}>
                    <button onClick={ cancel_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#707070'}}>Cancel</button>


                    {targetName.trim().length > 0 && previewURL.length > 0 && targetSpec_cpu.trim().length > 0 && targetSpec_gpu.trim().length > 0 && targetSpec_memory.trim().length > 0 && targetSpec_model.trim().length > 0 ?
                        <button onClick={ create_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Create</button>

                    :
                        <button style={{width:'95px', height:'35px', float:'right', backgroundColor:'lightgrey', marginRight:'15px'}} disabled>Create</button>

                    }

                    {/*
                    <button onClick={ create_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Create</button>
                    */}
                </div>
            </div>
        </div>
    );
}


/* 프로젝트 이름 수정 팝업 */
export function TargetModifyPopup({
    targetName, setTargetName,
    targetImage, previewURL,
    targetSpec_cpu, targetSpec_gpu, targetSpec_memory, targetSpec_model,
    specChange, uploadImageFile,
    cancel_ButtonClick,
    apply_ButtonClick})
{
    return (
        <div className='popup' id='modify_target_popup'>
            <div className="popup_inner">
                <div className="popup_header" style={{textAlign:'left', fontSize:'16px', color:'#6c7890', fontWeight:'bold'}}>Modify Target</div>

                <div className="popup_body">
                    <label className="popup_label" style={{textAlign:'left', margin:'0px'}}>Name</label>
                    <input
                        id='input_target_name'
                        value={targetName}
                        onChange={({ target: { value } }) => setTargetName(value)}
                        type="text"
                        placeholder="Target Name"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'20px'}}
                        maxLength='30'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'0px'}}>Image</label>

                    <div className="target_filebox">

                        <label htmlFor="input_target_image">파일 찾기</label>
                        <input
                            id='input_target_image'
                            /* value={targetImage} */
                            onChange={uploadImageFile}
                            type="file"
                            accept='image/jpg,image/png,image/jpeg'
                            style={{padding:'0px 0px 0px 10px', backgroundColor:'red'}}
                        />
                    </div>

                    <div style={{width:'100%', height:'180px', backgroundColor:'#303030', marginBottom:'30px', textAlign:'center'}}>
                        {previewURL !== '' &&
                            <img src={previewURL} alt='preview' style={{width:'240px', height:'180px', textAlign:'center'}}/>
                         }
                    </div>

                <div className="popup_header" style={{textAlign:'left', fontSize:'14px', color:'#6c7890', fontWeight:'bold', marginBottom:'20px'}}>Specification</div>

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>CPU</label>
                    <input
                        id='input_target_cpu'
                        value={targetSpec_cpu}
                        onChange={({ target: { value } }) => specChange(1, value)}
                        type="text"
                        placeholder="CPU"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>GPU</label>
                    <input
                        id='input_target_gpu'
                        value={targetSpec_gpu}
                        onChange={({ target: { value } }) => specChange(2, value)}
                        type="text"
                        placeholder="GPU"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>Memory</label>
                    <input
                        id='input_target_memory'
                        value={targetSpec_memory}
                        onChange={({ target: { value } }) => specChange(3, value)}
                        type="text"
                        placeholder="Memory"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'10px'}}>Model</label>
                    <input
                        id='input_target_model'
                        value={targetSpec_model}
                        onChange={({ target: { value } }) => specChange(4, value)}
                        type="text"
                        placeholder="Model"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='100'
                    />
                </div>

                <div className="popup_button_list" style={{width:'100%'}}>
                    <button onClick={ cancel_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#707070'}}>Cancel</button>

                    {targetName.trim().length > 0 && previewURL.length > 0 && targetSpec_cpu.trim().length > 0 && targetSpec_gpu.trim().length > 0 && targetSpec_memory.trim().length > 0 && targetSpec_model.trim().length > 0 ?
                        <button onClick={ apply_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Modify</button>
                    :
                        <button style={{width:'95px', height:'35px', float:'right', backgroundColor:'lightgrey', marginRight:'15px'}} disabled>Modify</button>
                    }

                    {/*
                    <button onClick={ apply_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Apply</button>
                    */}
                </div>
            </div>
        </div>
    );
}


