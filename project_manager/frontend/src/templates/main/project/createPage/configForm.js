import React from "react";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

import '../../../../CSS/project_management.css'

import Select from 'react-select';

function ConfigForm(
    {inputMethodList, outputMethodList, userEditList,
    target, taskType,
    datasetFile, baseModel,
    nasType, setNasType,
    weightLevel, setWeightLevel,
    precisionLevel, setPrecisionLevel,
    processingLib, setProcessingLib,
    userEdit, setUserEdit,
    inputMethod, setInputMethod,
    inputDataPath, setInputDataPath,
    outputMethod, setOutputMethod})
{

//    useEffect(() =>
//    {
//        console.log("config Form")
//    }, []);

    return (
        <>
        <div className="project_user_requirement" style={{border:'5px solid #303030'}}>

            {/* Task 선택 */}
            <div className='project_user_requirement_task_type' style={{height:'auto', borderBottom:'3px solid #303030'}}>
                <div style={{display:"flex", width:'100%', height:'100%'}}>
                    <div style={{width:'20%', minWidth:'150px', backgroundColor:'#707070', textAlign:'center',
                                 padding:'10px 10px 10px 10px'}}>

                        <div style={{padding:'0px 20px 0px 20px', color:'white'}}>Task Type</div>
                    </div>

                    <div style={{width:'80%', display:'flex', padding:'10px 10px 10px 10px'}}>
                        <input type="radio" name="task_type_radio" value="classification" style={{marginLeft:'20px'}}
                            checked={taskType === 'classification'} readOnly/>
                            <span style={{fontSize:'16px'}}>Classification</span>
                        <input type="radio" name="task_type_radio" value="detection" style={{marginLeft:'20px'}}
                            checked={taskType === 'detection'} readOnly/>
                            <span style={{fontSize:'16px'}}>Detection</span>
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
            <div className='project_user_requirement_nas_type' style={{height:'auto'}}>
                <div style={{display:"flex", width:'100%', height:'100%'}}>
                    <div style={{width:'20%', minWidth:'150px', backgroundColor:'#707070', textAlign:'center', padding:'10px 10px 10px 10px'}}>
                        <div style={{padding:'0px 20px 0px 20px', color:'white'}}>Nas Type</div>
                    </div>

                    <div style={{width:'80%', display:'flex', padding:'10px 10px 10px 10px'}}>
                        <input type="radio" name="nas_type_radio" value="bb_nas"
                            onChange={({ target: { value } }) => setNasType(value)} style={{marginLeft:'20px'}}
                            checked={nasType === 'bb_nas'}/>
                            <span style={{fontSize:'16px'}}>Backbone Nas</span>
                        <input type="radio" name="nas_type_radio" value="neck_nas"
                            onChange={({ target: { value } }) => setNasType(value)} style={{marginLeft:'20px'}}
                            checked={nasType === 'neck_nas'}/>
                            <span style={{fontSize:'16px'}}>Neck Nas</span>
                    </div>
                </div>
            </div>

            {/* Deploy Configuration */}
            <div className='project_user_requirement_deploy_config' style={{height:'auto', borderTop:'3px solid #303030', display:target.info === 'pc' || target.info === 'cloud' ? 'block' : 'none'}}>

                <div style={{display:"grid", width:'100%', height:'100%', gridTemplateColumns:'auto 80%', gridTemplateRows:'1fr 1fr'}}>

                    <div style={{gridRow:'1/3', gridColumn:'1/2', minWidth:'150px', backgroundColor:'#707070', textAlign:'center', padding:'10px 10px 10px 10px'}}>
                        <div style={{padding:'0px 20px 0px 20px', color:'white', alignItems:'center', display:'inline-flex', height:'100%'}}>Deploy Config</div>
                    </div>

                    <div className='deploy-config' style={{gridRow:'1/2', gridColumn:'2/3'}}>
                        <div style={{width:'100%', display:'grid', gridTemplateColumns:'1fr 1fr 1fr 1fr', padding:'10px 10px 10px 20px'}}>
                            <div style={{ gridColumn:'1/2'}}>
                                <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Light Weight Level</label>
                                <input
                                    className="config-input" type="number" min="0" max="10" step="0" maxLength={10}
                                    style={{padding:'0px 10px 0px 10px', width:'80%'}}
                                    value={weightLevel}
                                    onChange={({ target: { value } }) => setWeightLevel(value)}
                                    onKeyDown={(evt) => evt.key && evt.preventDefault()}/>
                            </div>

                            <div style={{ gridColumn:'2/3'}}>
                                <label style={{textAlign:'right', width:'30%', fontSize:'0.8rem'}}>Precision Level</label>
                                <input
                                    className="config-input" type="number" min="0" max="10" step="0" maxLength={2}
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
        </>
    );
}

export default ConfigForm;