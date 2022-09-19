import Cookies from "universal-cookie";

/* 웹 브라우저 쿠키 정보 유무 확인 */
export const isToken = () => new Cookies().get('TANGO_TOKEN')
export const isUser = () => new Cookies().get('userinfo')

/* 사용자 인증 방법 변경 */
//export const isToken = () =>  sessionStorage.getItem("TANGO_TOKEN");
//export const isUser = () =>  sessionStorage.getItem("userinfo");