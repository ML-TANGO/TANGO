import React from "react"
import { useEffect } from "react";
import { isToken, isUser } from '../../../Router/isLogin';

import Main_Image from "../../../images/Tango_main.png";

function Home()
{
    useEffect(() =>
    {
        if(  !!isToken() === false || !!isUser() === false )
        {
            /* 로그인 상태가 아닌 경우  */
            window.location.replace("/");
        }
    }, []);


    /* 홈 화면 - 메인 이미지 컴포넌트 */
    return (
        <div style={{width:'100%', height:'100%', textAlign:'center', position:'relative'}}>
            <img src={Main_Image} alt="main" style={{margin:'0 auto', position:'absolute', width:'40vw', top: '50%', left: '50%', transform: 'translate(-50%, -50%)'}} />
        </div>
    );
}

export default Home;
