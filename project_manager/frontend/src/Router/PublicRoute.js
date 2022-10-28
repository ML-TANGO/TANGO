import React from "react";
import { Navigate } from "react-router-dom";

//import isLogin from './isLogin';
import { isToken, isUser } from './isLogin';

//import Cookies from "universal-cookie";

function PublicRoute( { children, restricted } )
{
     /* 로그인이 되어있는 상태에서는 접근 불가 처리 - 로그인, 회원 가입 페이지*/
    if(  !!isToken() === false || !!isUser() === false )
    {
        /* 로그인 상태가 아닌 경우 - 로그인 및 회원 가입 페이지 접근 가능 */
        return children
    }
    else
    {
        /* 로그인 상태인 경우 - 메인 페이지로 이동 */
        return <Navigate to="/" />
    }
}

export default PublicRoute;