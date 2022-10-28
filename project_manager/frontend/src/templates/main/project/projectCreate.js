import React from "react";
import { useEffect, useState } from "react";

import { useLocation } from "react-router-dom";
//import { ChevronRight } from "react-bootstrap-icons";

import '../../../CSS/PMRouter.css'
import '../../../CSS/project_management.css'

import { isToken, isUser } from '../../../Router/isLogin';
import NeuralAndLoadPage from "./createPage/neuralAndLoadPage";

import * as Request from "../../../service/restProjectApi";

function ProjectCreate()
{
    // 프로젝트 메인 페이지로부터 전달 받은 파라미터 state
    const {state} = useLocation();

    const [project_id, setProject_id] = useState('');                                   // 프로젝트 ID
    const [project_name, setProject_name] = useState('');                               // 프로젝트 이름
    const [project_description, setProject_description] = useState('');                 // 프로젝트 설명

    // 현재 페이지 정보가 변경될 경우 반복 호출
    useEffect(() =>
    {
        if(  !!isToken() === false || !!isUser() === false )
        {
            /* 로그인 상태가 아닌 경우  */
            window.location.replace("/");
        }

        /* 전달 받은 파라미터가 있는 경우 */
        if(state === null)
        {
            alert('비정상적인 접근입니다.')
            window.location.replace("/");
        }
        else
        {
            setProject_id(state.id);
            setProject_name(state.name);
            setProject_description(state.description);
        }
    }, []);

    const description_save_button_click = () => {
        const description = project_description.trim();
        if(description.length > 0)
        {
            const data = {
                id:project_id,
                description:project_description
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

    return (
        <div className='manage_container'>
            {/* 프로젝트 생성 페이지 - 헤더 */}
            <div className='manage_header' style={{width:'100%'}}>
                <div className='title'>
                    <div className='path'></div>

                    <div className='title_left'>{ project_name }</div>

                </div>

                <div className="project_description" style={{marginTop:'10px', marginBottom:'10px'}}>
                   <div className="description-content" style={{ padding:'0px 20px 0px 20px', height:'40px', display:'flex'}}>
                        <span style={{color:'white'}}>Description</span>
                        <input onChange={({ target: { value } }) => setProject_description(value)} value={project_description} style={{ height:'30px', borderRadius:'5px', marginLeft:'10px', marginRight:'10px', fontSize:'16px'}} />
                        <button onClick={() => description_save_button_click()} style={{ height:'30px', width:'150px', borderRadius:'5px', backgroundColor:'#707070', color:'white', fontSize:'16px', border:'0px'}}>설명 수정</button>
                    </div>
                </div>
            </div>

            <div className='manage_bottom'>
                <div className='manage_bottom_component'>
                    {/* 신경망 생성 페이지 */}
                    <NeuralAndLoadPage project_id={state.id} project_name={state.name} project_description={state.description} />
                </div>
            </div>
        </div>
    );
}

export default ProjectCreate;