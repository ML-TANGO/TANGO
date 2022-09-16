import React from "react";
import { useState  } from "react";
import { Link } from "react-router-dom";

import * as Request from "../../service/restAuthApi";

import '../../CSS/auth.css';

import LoginImage from "../../images/Tango_login.png";
import Logo_Image from "../../images/logo_3.png";

function SignUp()
{
    const [duplicateCheck,setDuplicateCheck] = useState(false);                     /* 아이디 중복 검사 핸들러 */

    const [id,setID] = useState('');                                                /* 사용자 입력 아이디 */
    const [email,setEmail] = useState('');                                          /* 사용자 입력 이메일 */
    const [password,setPassword] = useState('');                                    /* 사용자 입력 패스워드 */
    const [passwordConfirm,setPasswordConfirm] = useState('');                      /* 사용자 입력 패스워드 확인 */

    const [idError,setIDError] = useState(false);                                   /* 입력 아이디 오류 */
    const [emailError,setEmailError] = useState(false);                             /* 입력 이메일 오류 */
    const [passwordError,setPasswordError] = useState(false);                       /* 입력 패스워드 오류 */
    const [passwordConfirmError,setPasswordConfirmError] = useState(false);         /* 입력 패스워드-확인 오류 */

    const handleClick = () => {
        if(idError || emailError || passwordError || passwordConfirmError)
        {
            alert("입력 정보를 다시 확인 해주세요");
            return;
        }

        if(id === '' || email === '' || password === '' || passwordConfirm === '')
        {
            alert("빈칸을 작성해주세요");
            return;
        }

        if(duplicateCheck === false)
        {
            alert("ID 중복 검사를 해주세요");
            return ;
        }

        try
        {
            const obj = {
                id : id,
                email : email,
                password : password
            };

            Request.requestSignUp(obj).then(result =>
            {
                alert("회원가입 성공")

                setDuplicateCheck(false);
                setID('');
                setEmail('');
                setPassword('');
                setPasswordConfirm('');

                window.location.replace("/");
            })
            .catch(error =>
            {
                alert('Signup Error')
            });
        } catch (err)
        {
            console.log(err);
        }
    };

     const idCheck = ()=>{
        if(id === '')
        {
            alert("ID를 작성 해주세요")
            return
        }

        if(idError)
        {
            alert("ID를 다시 작성 해주세요")
            return
        }
        try
        {

            Request.requestSignUpDuplicate(id).then(result =>
            {
                if(result.data['result'] === true)
                {
                    alert("사용 가능한 ID 입니다.")
                    setDuplicateCheck(true);
                }
                else
                {
                    alert("중복된 ID 입니다.")
                    setID('')
                }
            })
            .catch(error =>
            {
                alert('ID Check Error')
            });

        }
        catch (err)
        {
            console.log(err);
        }
    }


    const appKeyPress = (e) => {
        if (e.key === "Enter")
        {
            handleClick();
        }
    };

    const onChangeID = (e) =>{
        const userIdRegex = /^[A-Za-z0-9+]{5,20}$/;

        if ((!e.target.value || (userIdRegex.test(e.target.value)))) setIDError(false);
        else setIDError(true);
        setID(e.target.value);
        setDuplicateCheck(false);
    }

    const onChangeEmail= (e) =>{
        const emailRegex = /^([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?)$/i;

        if ((!e.target.value || (emailRegex.test(e.target.value)))) setEmailError(false);
        else setEmailError(true);
        setEmail(e.target.value);
    }

    const onChangePassword= (e) =>{
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*[$@$!%*?&])[A-Za-z\d$@$!%*?&]{8,}/;

        if ((!e.target.value || (passwordRegex.test(e.target.value)))) setPasswordError(false);
        else setPasswordError(true);
        setPassword(e.target.value);
    }

    const onChangePasswordConfirm = (e) =>{

        if (password === e.target.value) setPasswordConfirmError(false);
        else setPasswordConfirmError(true);
        setPasswordConfirm(e.target.value);
    }

    return (
        <div className="login">

            {/* 회원가입 페이지 좌측 그리드 */}
            <div className="login_left">
                <div className="top">
                    <img src={Logo_Image} alt="로고 이미지"/>&nbsp;
                    {/*<span style={{ color: "#448FFF", fontSize: "2vw"}}>Deep</span>
                    <span style={{ color: "#303030", fontSize: "2vw"}}>Framework</span>*/}

                    <span style={{ color: "#448FFF", fontSize: "3rem", fontWeight:"bold", marginLeft:"5px"}}>TANGO</span>
                </div>

                <div className="bottom">
                    <img src={LoginImage} alt="로그인 이미지"></img>
                </div>
            </div>

            {/* 회원가입 페이지 우측 그리드 */}
            <div className="login_right">
                <div className="login_form">
                    <div className="login-header" style={{textAlign:'center', fontWeight:"bold", color:"#303030"}}>Create Account</div>

                    {/* 회원가입 입력 폼 */}
                    <div className="login-body">

                        {/* 회원가입 입력 폼 - 아이디 중복확인 */}
                        <div style={{width:'100%', marginBottom:'5px'}}>
                            <input  value={id} onChange={onChangeID} type="text" placeholder='ID' style={{width:'75%'}} onKeyPress={appKeyPress} />

                            <button onClick={idCheck}  style={{ width:'20%', 'marginTop':'5px', 'marginLeft':'15px', height:'55px', fontSize:'15px', fontWeight:'400', float:'right'}}>Check</button>

                            <p style={{fontSize:'12px', color: idError ? '#303030' : 'transparent'}}>5~20자의 영문 소문자, 숫자와 특수기호(_),(-)만 사용 가능합니다.</p>
                        </div>

                        {/* 회원가입 입력 폼 - 이메일 입력 */}
                        <input value={email} onChange={onChangeEmail} type='email' placeholder='E-Mail' onKeyPress={appKeyPress} />

                        <p style={{fontSize:"12px", color: emailError ? "#303030" : "transparent"}}>이메일 형식이 잘못되었습니다</p>

                        {/* 회원가입 입력 폼 - 패스워드 입력 */}
                        <input value={password} onChange={onChangePassword} type='password' placeholder='Password' onKeyPress={appKeyPress} />

                        <p style={{fontSize:"12px", color: passwordError ? "#303030" : "transparent"}}>8자 이상, 영문 대소문자, 특수문자를 사용하세요</p>

                        {/* 회원가입 입력 폼 - 패스워드 확인 입력 */}
                        <input style={{}} value={passwordConfirm} onChange={onChangePasswordConfirm} type="password" placeholder="PasswordConfirm" onKeyPress={appKeyPress} />

                        <p style={{fontSize:"12px", color: passwordConfirmError ? "#303030" : "transparent"}}>비밀번호가 일치하지 않습니다.</p>

                        {/* 회원가입 입력 폼 - 버튼 ( 생성, 뒤로 ) */}
                        <div className="login-button" style={{'marginTop':'20px'}}>

                            {duplicateCheck === true && passwordError === false && passwordConfirmError === false && password.length > 0 &&  passwordConfirm.length > 0 ?
                                <button onClick={handleClick}>Create Account</button>
                            :
                                <button style={{backgroundColor:'lightgrey'}}>Create Account</button>
                            }

                           <Link to="/login" >
                                <button style={{'marginTop':'15px', fontWeight: "400", 'backgroundColor':'#707070'}}>Cancel</button>
                            </Link>
                        </div>
                    </div>


                </div>
            </div>
        </div>
    );
}

export default SignUp;