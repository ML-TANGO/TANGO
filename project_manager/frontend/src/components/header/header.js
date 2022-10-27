import React from "react";
import { useEffect, useState } from "react";

//import { useDispatch } from "react-redux";
//import { setUserID } from "../../actions";

//import * as Request from "../../service/restAuthApi";
//import * as BluRequest from "../../service/rest_BlueAiApi";

import Cookies from "universal-cookie";
import { isToken, isUser } from '../../Router/isLogin';

import { List } from "react-bootstrap-icons";
//import { Link } from "react-router-dom";

import '../../CSS/header.css'

export default function Header()
{
    const [id, setId] = useState('');

    useEffect( () =>
    {
        const cookies = new Cookies();
        var user_info = cookies.get('userinfo')
        setId(user_info)
    }, []);

//    const dispatch = useDispatch();

    /* 설정 아이콘 클릭 이벤트 */
    function setting_icon_Click()
    {
         console.log('setting_icon_Click')
    }

    /* 알림 아이콘 클릭 이벤트 */
    function notify_icon_Click()
    {
        console.log('notify_icon_Click')
    }

    /* 사용자 아이콘 클릭 이벤트 */
    function logout()
    {
         console.log('user_icon_Click')

        var cookies = new Cookies();
        cookies.remove('userinfo', {path:'/'});
        cookies.remove('TANGO_TOKEN', {path:'/'});

        if(  !!isToken() === false || !!isUser() === false )
        {
            /* 로그인 상태가 아닌 경우  */
            window.location.replace("/");
        }
    }

    return (
        <>
        <div style={{position:'relative', height:'100%', width:'100%', zIndex:'1001'}}>

            <div className='header_menu' style={{height:'100%', width:'10%', float:'left'}}>
                <List color="#C2C8D3" size="30px" className="burger-menu"/>

                <div className="header-menu-list" style={{ height:'auto', width:'100%'}}>

                    <a href="/">Home</a>
                    <a href="/project">Project Management</a>
                    <a href="/target">Target Management</a>
                    <a href="/data">Data Management</a>
                    <a href="/visualization">Visualization</a>

                </div>
            </div>

            <div id='header_item' style={{height:'100%', width:'50%', float:'right'}}>

                <div className="icon-bar">
                    {/* 설정 아이콘 */}
                    <div className="icon_setting">
                        <div id='icon_setting_img' onClick={ setting_icon_Click }></div>
                    </div>

                    {/* 알림 아이콘 */}
                    <div className="icon_notify">
                        <div id='icon_notify_img' onClick={ notify_icon_Click }></div>
                    </div>

                    {/* 사용자 아이디 */}
                    {/* <div className="icon_user" onClick={ user_icon_Click }>
                        <span id="user_id">{ id }</span>
                    </div> */}

                    <div className="logout-dropdown icon_user">
                      <span id="user_id">{ id }</span>
                      <div className="logout-dropdown-content">
                        <a onClick={ logout }>Logout</a>
                      </div>
                    </div>
                </div>
            </div>
        </div>


        </>
    );
}
