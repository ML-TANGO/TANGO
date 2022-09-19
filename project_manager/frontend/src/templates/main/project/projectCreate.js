import React from "react";
import { useEffect, useState } from "react";

import { useLocation } from "react-router-dom";
//import { ChevronRight } from "react-bootstrap-icons";

import '../../../CSS/PMRouter.css'
import '../../../CSS/project_management.css'

import { isToken, isUser } from '../../../Router/isLogin';
import NeuralAndLoadPage from "./createPage/neuralAndLoadPage";

function ProjectCreate()
{
    // 프로젝트 메인 페이지로부터 전달 받은 파라미터 state
    const {state} = useLocation();

//    const [project_id, setProject_id] = useState('');                                   // 프로젝트 ID
    const [project_name, setProject_name] = useState('');                               // 프로젝트 이름
//    const [project_description, setProject_description] = useState('');                 // 프로젝트 설명

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
//            setProject_id(state.id);
            setProject_name(state.name);
        }
    }, []);

    return (
        <div className='manage_container'>
            {/* 프로젝트 생성 페이지 - 헤더 */}
            <div className='manage_header' style={{width:'100%'}}>
                <div className='title'>
                    <div className='title_left'>{ project_name }</div>
                    <div className='title_right'></div>
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