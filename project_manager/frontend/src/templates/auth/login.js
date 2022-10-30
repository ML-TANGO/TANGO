import React, { useState } from "react";
import { Link } from "react-router-dom";

import Cookies from "universal-cookie";

import * as Request from "../../service/restAuthApi";
//import * as BluRequest from "../../service/rest_BlueAiApi";

import '../../CSS/auth.css';

import LoginImage from "../../images/Tango_login.png";
import Logo_Image from "../../images/logo_3.png";

function Login()
{
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");

    const appKeyPress = (e) => {
        if (e.key === "Enter")
        {
            loginButtonClick();
        }
    };

    /* 로그인 버튼 클릭 */
    function loginButtonClick()
    {
        Request.requestLogin(email, password).then(result =>
        {
            var user_exist = result.data['result']

            if(user_exist === true)
            {
                var content = JSON.parse(result.data['content'])

                var cookies = new Cookies();
                cookies.set('TANGO_TOKEN', content['access_token'], {path:"/"});
                cookies.set('userinfo', email, {path:"/"});

                window.location.replace("/");
            }
            else
            {
                alert('사용자 정보를 확인해주세요')
            }

        }).catch(error =>
        {
            alert('login error')
        });


    };

    return (
        <div className="login">

            {/* 로그인 페이지 좌측 그리드 */}
            <div className="login_left">
                <div className="top" style={{position:"relative"}}>
                    <img src={Logo_Image} alt="로고 이미지"/>&nbsp;
                    {/*
                    <span style={{ color: "#448FFF", fontSize: "2vw"}}>Deep</span>
                    <span style={{ color: "#303030", fontSize: "2vw"}}>Framework</span>*/}

                    <span style={{ color: "#448FFF", fontSize: "3rem", fontWeight:"bold", marginLeft:"5px"}}>
                        TANGO
                    </span>
                </div>

                <div className="bottom">
                    <img src={LoginImage} alt="로그인 이미지"></img>
                </div>
            </div>

            {/* 로그인 페이지 우측 그리드 */}
            <div className="login_right">
                <div className="login_form">
                    <div className="login-header" style={{textAlign:'center', fontWeight:"bold", color:"#303030"}}>
                        TANGO
                    </div>

                    {/* 로그인 입력 폼 - 아이디, 패스워드 */}
                    <div className="login-body">
                        <input
                            value={email}
                            onChange={({ target: { value } }) => setEmail(value)}
                            type="text"
                            placeholder="ID"
                            onKeyPress={appKeyPress}
                        />
                        <input
                            value={password}
                            onChange={({ target: { value } }) => setPassword(value)}
                            type="password"
                            placeholder="Password"
                            onKeyPress={appKeyPress}
                        />
                    </div>

                    {/* 로그인 입력 폼 버튼 - 로그인, 회원가입 */}
                    <div className="button_list" style={{'marginTop':'20px'}}>
                        <button onClick={ loginButtonClick }>Log in</button>

                        {/*
                        <Link to="/signup" >
                            <button style={{fontWeight: "400", 'backgroundColor':'#707070'}}>Sign Up</button>
                        </Link>
                        */}

                        <div style={{marginTop:'10px', fontSize:'14px'}}>
                            <span>Don{'\''}t have account?</span>
                            <span style={{marginLeft:'5px', color:'4A80FF'}}>
                                <Link className='sign_up_link' to="/signup" style={{color:'4A80FF'}}>
                                    Register now
                                </Link>
                            </span>
                        </div>

                    </div>

                </div>
            </div>
        </div>
    );
}

export default Login;