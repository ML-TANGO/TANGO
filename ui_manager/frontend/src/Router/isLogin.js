import Cookies from "universal-cookie";

/* 웹 브라우저내에 쿠키 정보 유무 확인 */
export const isToken = () => new Cookies().get('DF_TOKEN')
export const isUser = () => new Cookies().get('userinfo')

/* 20220304 jpchoi - 사용자 인증 방법 변경 */
//export const isToken = () =>  sessionStorage.getItem("DF_TOKEN");
//export const isUser = () =>  sessionStorage.getItem("userinfo");