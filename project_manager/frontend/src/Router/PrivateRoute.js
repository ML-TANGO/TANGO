import React from "react";
import { Navigate } from "react-router-dom";
import { isToken, isUser } from './isLogin';
//import Cookies from "universal-cookie";


function PrivateRoute( { children, restricted } )
{
    if( !!isToken() === false || !!isUser() === false )
    {
        /* 메인 페이지 루트 - 로그인 상태가 아닌 경우 로그인 페이지로 이동 */
        return <Navigate to="/login" />
    }
    else
    {
        /* 메인 페이지 루트 - 메인 페이지 이동 */
        return children
    }

}
export default PrivateRoute;

