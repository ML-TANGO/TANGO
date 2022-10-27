import React from "react";
import { useEffect } from "react";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";

import "../../CSS/sidebar.css";

import Sidebar_main_image from "../../images/logo_3.png";

import project_img_on from "../../images/icons/icon_3x/project_mgmt_on@3x.png";
import project_img_off from "../../images/icons/icon_3x/project_mgmt.png";

import target_img_on from "../../images/icons/icon_3x/target_on@3x.png";
import target_img_off from "../../images/icons/icon_3x/target@3x.png";

import data_img_on from "../../images/icons/icon_3x/data_mgmt_on@3x.png";
import data_img_off from "../../images/icons/icon_3x/data_mgmt@3x.png";

import visual_img_on from "../../images/icons/icon_3x/visualization_on.png";
import visual_img_off from "../../images/icons/icon_3x/visualization.png";

//import * as Request from "../../service/restAuthApi";

//import Cookies from "universal-cookie";
//import { isToken, isUser } from '../../Router/isLogin';



function Sidebar()
{
    const navigate = useNavigate();

    /* 새로고침 이후 사이드바 버튼 포커싱 재설정 */
    useEffect(() =>
    {
        var path_name = window.location.pathname.replace('/','')

        if( path_name === '' )
        {
            home_buttonClick()
        }
        else if( path_name.includes('project') )
        {
            project_buttonClick()
        }
        else if( path_name.includes('target') )
        {
            target_buttonClick()
        }
        else if( path_name.includes('data') )
        {
            data_buttonClick()
        }
        else if( path_name.includes('visualization') )
        {
            visual_buttonClick()
        }
    }, []);

    /* 로그아웃 이벤트 */
//    const logout = () =>
//    {
//        var cookies = new Cookies();
//        cookies.remove('userinfo', {path:'/'});
//        cookies.remove('TANGO_TOKEN', {path:'/'});
//
//        if(  !!isToken() === false || !!isUser() === false )
//        {
//            /* 로그인 상태가 아닌 경우  */
//            window.location.replace("/");
//        }
//    }

    /* 사이드 바 - 홈 버튼 클릭 이벤트 */
    const home_buttonClick = () =>
    {
        var item_1 = document.getElementById('item_1');
        var item_2 = document.getElementById('item_2');
        var item_3 = document.getElementById('item_3');
        var item_4 = document.getElementById('item_4');

        item_1.style.backgroundColor = '';
        item_1.style.color = '#6c7890';

        item_2.style.backgroundColor = '';
        item_2.style.color = '#6c7890';

        item_3.style.backgroundColor = '';
        item_3.style.color = '#6c7890';

        item_4.style.backgroundColor = '';
        item_4.style.color = '#6c7890';

        document.getElementById('project_button_icon').style.backgroundImage = "url('" + project_img_off + "')";
        document.getElementById('target_button_icon').style.backgroundImage = "url('" + target_img_off + "')";
        document.getElementById('data_button_icon').style.backgroundImage = "url('" + data_img_off + "')";
        document.getElementById('visual_button_icon').style.backgroundImage = "url('" + visual_img_off + "')";
    }

    /* 사이드 바 - 프로젝트 관리 버튼 클릭 이벤트 */
    const project_buttonClick = () =>
    {
        var item_1 = document.getElementById('item_1');
        var item_2 = document.getElementById('item_2');
        var item_3 = document.getElementById('item_3');
        var item_4 = document.getElementById('item_4');

        item_1.style.backgroundColor = '#4a80ff';
        item_1.style.color = '#ffffff';
        document.getElementById('project_button_icon').style.backgroundImage = "url('" + project_img_on + "')";

        item_2.style.backgroundColor = '';
        item_2.style.color = '#6c7890';
        document.getElementById('target_button_icon').style.backgroundImage = "url('" + target_img_off + "')";

        item_3.style.backgroundColor = '';
        item_3.style.color = '#6c7890';
        document.getElementById('data_button_icon').style.backgroundImage = "url('" + data_img_off + "')";

        item_4.style.backgroundColor = '';
        item_4.style.color = '#6c7890';
        document.getElementById('visual_button_icon').style.backgroundImage = "url('" + visual_img_off + "')";
    }

    /* 사이드 바 - 타겟 관리 버튼 클릭 이벤트 */
    const target_buttonClick = () =>
    {
        var item_1 = document.getElementById('item_1');
        var item_2 = document.getElementById('item_2');
        var item_3 = document.getElementById('item_3');
        var item_4 = document.getElementById('item_4');

        item_1.style.backgroundColor = '';
        item_1.style.color = '#6c7890';
        document.getElementById('project_button_icon').style.backgroundImage = "url('" + project_img_off + "')";

        item_2.style.backgroundColor = '#4a80ff';
        item_2.style.color = '#ffffff';
        document.getElementById('target_button_icon').style.backgroundImage = "url('" + target_img_on + "')";

        item_3.style.backgroundColor = '';
        item_3.style.color = '#6c7890';
        document.getElementById('data_button_icon').style.backgroundImage = "url('" + data_img_off + "')";

        item_4.style.backgroundColor = '';
        item_4.style.color = '#6c7890';
        document.getElementById('visual_button_icon').style.backgroundImage = "url('" + visual_img_off + "')";
    }


    /* 사이드 바 - 데이터 관리 버튼 클릭 이벤트 */
    const data_buttonClick = () =>
    {
        var item_1 = document.getElementById('item_1');
        var item_2 = document.getElementById('item_2');
        var item_3 = document.getElementById('item_3');
        var item_4 = document.getElementById('item_4');

        item_1.style.backgroundColor = '';
        item_1.style.color = '#6c7890';
        document.getElementById('project_button_icon').style.backgroundImage = "url('" + project_img_off + "')";

        item_2.style.backgroundColor = '';
        item_2.style.color = '#6c7890';
        document.getElementById('target_button_icon').style.backgroundImage = "url('" + target_img_off + "')";

        item_3.style.backgroundColor = '#4a80ff';
        item_3.style.color = '#ffffff';
        document.getElementById('data_button_icon').style.backgroundImage = "url('" + data_img_on + "')";

        item_4.style.backgroundColor = '';
        item_4.style.color = '#6c7890';
        document.getElementById('visual_button_icon').style.backgroundImage = "url('" + visual_img_off + "')";
    }

    /* 사이드 바 - visualization 버튼 클릭 이벤트 */
    const visual_buttonClick = () =>
    {
        var item_1 = document.getElementById('item_1');
        var item_2 = document.getElementById('item_2');
        var item_3 = document.getElementById('item_3');
        var item_4 = document.getElementById('item_4');

        item_1.style.backgroundColor = '';
        item_1.style.color = '#6c7890';
        document.getElementById('project_button_icon').style.backgroundImage = "url('" + project_img_off + "')";

        item_2.style.backgroundColor = '';
        item_2.style.color = '#6c7890';
        document.getElementById('target_button_icon').style.backgroundImage = "url('" + target_img_off + "')";

        item_3.style.backgroundColor = '';
        item_3.style.color = '#6c7890';
        document.getElementById('data_button_icon').style.backgroundImage = "url('" + data_img_off + "')";

        item_4.style.backgroundColor = '#4a80ff';
        item_4.style.color = '#ffffff';
        document.getElementById('visual_button_icon').style.backgroundImage = "url('" + visual_img_on + "')";
    }

    return (
        <>
        <div className="sidebar_menu" style={{width:'300px'}}>
            <div className="sidebar_list" style={{width:'100%'}}>

                {/* 사이드바 아이템 - 메인 */}
                <div className="sidebar_main" style={{padding:"25px 0px", justifyContent:'center'}}>
                    <Link to="/" onClick={ home_buttonClick }>
                        <img src={ Sidebar_main_image } alt="TANGO 로고 이미지 입니다." style={{ width:"75px", height:"75px"}}></img>
                        <br/>

                        {/*<span style={{ color: "#448FFF" }}>Deep</span>
                        <span style={{ color: "white" }}>Framework</span>*/}

                        <span style={{ color: "#448FFF", fontWeight:"bold", fontSize:"3rem"}}>TANGO</span>
                    </Link>
                </div>

                {/* 사이드바 아이템 - 프로젝트 관리, 데이터 관리, 로그아웃 */}
                <div className="sidebar_item">
                    <ul>
                        <li>
                            {/* 프로젝트 관리 페이지 이동 */}
                            <Link to="/project" onClick={ () =>project_buttonClick() }>
                                <div className='item' id='item_1'>
                                    <div id='project_button_icon'>
                                        <span className='item_text'>Project Management</span>
                                    </div>
                                </div>
                            </Link>
                        </li>

                        {/* 타겟 관리 페이지 이동 */}
                        <li>
                            <Link to="/target" onClick={ () => target_buttonClick() }>
                                 <div className='item' id='item_2'>
                                    <div id='target_button_icon'>
                                        <span className='item_text'>Target Management</span>
                                    </div>
                                </div>
                            </Link>
                        </li>

                        {/* 데이터 관리 페이지 이동 */}
                        <li>
                            <Link to="/data" onClick={ () => data_buttonClick() }>
                                 <div className='item' id='item_3'>
                                    <div id='data_button_icon'>
                                        <span className='item_text'>Data Management</span>
                                    </div>
                                </div>
                            </Link>
                        </li>

                       {/* 시각화 페이지 이동 */}
                        <li>
                            <Link to="/visualization" onClick={ () => visual_buttonClick() }>
                                 <div className='item' id='item_4'>
                                    <div id='visual_button_icon'>
                                        <span className='item_text'>Visualization</span>
                                    </div>
                                </div>
                            </Link>
                        </li>
                    </ul>

                </div>

            </div>
        </div>
        </>
    );
}

export default Sidebar;