import React from "react";
//import { useEffect, useState } from "react";

import '../../CSS/popup.css'

/* 프로젝트 생성 팝업 */
export function ProjectCreatePopup({projectName, setProjectName, projectDescription, setProjectDescription, cancel_ButtonClick, create_ButtonClick})
{
    return (
        <div className='popup' id='create_project_popup'>
            <div className="popup_inner">
                <div className="popup_header" style={{textAlign:'left', fontSize:'16px', color:'#6c7890', fontWeight:'bold'}}>Create Project</div>

                <div className="popup_body">
                    <label className="popup_label" style={{textAlign:'left', margin:'0px'}}>Name</label>
                    <input
                        id='input_project_name'
                        className="project_input"
                        value={projectName}
                        onChange={({ target: { value } }) => setProjectName(value)}
                        type="text"
                        placeholder="Project Name"
                        style={{padding:'0px 0px 0px 10px', marginBottom:'10px'}}
                        maxLength='30'
                    />

                    <label className="popup_label" style={{textAlign:'left', marginTop:'0px'}}>Description</label>
                    <input
                        id='input_project_description'
                        className="project_input"
                        value={projectDescription}
                        onChange={({ target: { value } }) => setProjectDescription(value)}
                        type="text"
                        placeholder="Project Description"
                        style={{padding:'0px 0px 0px 10px'}}
                        maxLength='100'
                    />
                </div>
                <div className="popup_button_list" style={{width:'100%'}}>
                    <button onClick={ cancel_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#707070'}}>Cancel</button>
                    <button onClick={ create_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Create</button>
                </div>
            </div>
        </div>
    );
}


/* 프로젝트 이름 수정 팝업 */
export function ProjectRenamePopup({ getProject, newProjectName, setNewProjectName, cancel_ButtonClick, apply_ButtonClick })
{
    return (
        <div className='popup' id='project_name_modify_popup'>
            <div className="popup_inner">
                <div className="popup_header" style={{textAlign:'left', fontSize:'16px', color:'#6c7890', fontWeight:'bold'}}>Rename Project</div>

                <div className="popup_body">
                    <label className="popup_label" style={{textAlign:'left', margin:'0px'}}>Enter New Project Name</label>
                    <input
                        id='input_modify_name'
                        className="project_input"
                        value={newProjectName}
                        onChange={({ target: { value } }) => setNewProjectName(value)}
                        type="text"
                        placeholder="Modify Project"
                        style={{padding:'0px 0px 0px 10px'}}
                        maxLength='30'
                    />
                </div>

                <div className="popup_button_list" style={{width:'100%'}}>
                    <button onClick={ cancel_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#707070'}}>Cancel</button>
                    <button onClick={ apply_ButtonClick } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Apply</button>
                </div>
            </div>
        </div>
    );
}



/* 프로젝트 생성 - 타겟 고급 설정 팝업 */
export function ProjectTargetSetting( { setTarget_detail } )
{

    function modify_ButtonClick()
    {
        const target_setting = {
            'target_os' :           document.getElementById('target_os').value,
            'target_engine' :       document.getElementById('target_engine').value,
            'dependency_module' :   document.getElementById('dependency_module').value,
            'min_latency' :         document.getElementById('min_latency').value,
            'max_electric' :        document.getElementById('max_electric').value,
            'max_modelSize' :       document.getElementById('max_modelSize').value,
            'max_memory' :          document.getElementById('max_memory').value,
        }

        // 타겟 고급 설정 변경
        setTarget_detail(target_setting);

        document.getElementById('project_target_modify_popup').style.display = 'none';
    }

    function cancel_ButtonClick()
    {
        document.getElementById('project_target_modify_popup').style.display = 'none';
    }

    return (
        <div className='popup' id='project_target_modify_popup'>
            <div className="target_popup_inner">
                <div className="target_popup_header" style={{textAlign:'left', fontSize:'16px', marginBottom:'20px', color:'#6c7890', fontWeight:'bold'}}>Target detail Info</div>

                <div className="target_popup_body" style={{float:'right'}}>
                    <label className="target_popup_label" style={{width:'50px'}}>OS : </label>
                    <input id='target_os' type="text" style={{width:'130px', padding:'0px 0px 0px 10px'}} maxLength='30' />
                </div>

                 <div className="target_popup_body" style={{float:'right'}}>
                    <label className="target_popup_label" style={{width:'50px'}}>Engine : </label>
                    <input id='target_engine' type="text" style={{width:'130px', padding:'0px 0px 0px 10px'}} maxLength='30' />
                 </div>

                  <div className="target_popup_body" style={{float:'right'}}>
                     <label className="target_popup_label" style={{width:'50px'}}>Dependency Module : </label>
                     <input id='dependency_module' type="text" style={{width:'130px', padding:'0px 0px 0px 10px'}} maxLength='30' />
                  </div>

                  <div className="target_popup_body" style={{float:'right'}}>
                     <label className="target_popup_label" style={{width:'50px'}}>Min Latency ( FPS ): </label>
                     <input id='min_latency' type="text" style={{width:'130px', padding:'0px 0px 0px 10px'}} maxLength='30' />
                  </div>

                  <div className="target_popup_body" style={{float:'right'}}>
                     <label className="target_popup_label" style={{width:'50px'}}>Max Consume Electric : </label>
                     <input id='max_electric' type="text" style={{width:'130px', padding:'0px 0px 0px 10px'}} maxLength='30' />
                  </div>

                  <div className="target_popup_body" style={{float:'right'}}>
                     <label className="target_popup_label" style={{width:'50px'}}>Max Model Size ( byte ) : </label>
                     <input id='max_modelSize' type="text" style={{width:'130px', padding:'0px 0px 0px 10px'}} maxLength='30' />
                  </div>

                  <div className="target_popup_body" style={{float:'right'}}>
                     <label className="target_popup_label" style={{width:'50px'}}>Max Memory : </label>
                     <input id='max_memory' type="text" style={{width:'130px', padding:'0px 0px 0px 10px'}} maxLength='30' />
                  </div>

                <div className="popup_button_list" style={{width:'100%'}}>
                    <button onClick={()=> cancel_ButtonClick() } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#2f343f'}}>Cancel</button>
                    <button onClick={()=> modify_ButtonClick() } style={{width:'95px', height:'35px', float:'right', backgroundColor:'#4a80ff', marginRight:'15px'}}>Modify</button>
                </div>
            </div>
        </div>
    );
}

