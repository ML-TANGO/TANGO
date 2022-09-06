import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

/* 로그인 인증 확인 */
import PublicRoute from "./Router/PublicRoute";
import PrivateRoute from "./Router/PrivateRoute";

/* 이동 페이지 정보 - 로그인 완료 후 페이지 루트 */
import Main from "./templates/main/main";
import NotFound from "./templates/NotFound";
import Home from "./templates/main/home/home";

/* 이동 페이지 정보 - 프로젝트 관리 및 생성 페이지 */
import ProjectMain from "./templates/main/project/projectMain";
import ProjectCreate from "./templates/main/project/projectCreate";

/* 이동 페이지 정보 - 데이터 셋 관리 및 [ 레이블링 저작도구 ] */
import DataMain from "./templates/main/data/dataMain";

/* 이동 페이지 정보 - 데이터 셋 관리 및 [ 레이블링 저작도구 ] */
import VisualMain from "./templates/main/visual/visualMain";

/* 이동 페이지 정보 - 로그인 및 회원 가입 */
import Login from "./templates/auth/login";
import SignUp from "./templates/auth/signup";

function App()
{
    return (
        <BrowserRouter>
            <Routes>
                {/* 로그인 후 페이지 - 홈, 프로젝트 및 데이터 셋 관리 */}
                <Route path="/" element={ <PrivateRoute> <Main/> </PrivateRoute> }>
                    <Route path="" element={ <Home/> } />
                    <Route path="project/*" element={ <ProjectMain/> }/>
                    <Route path="project/create/" element={ <ProjectCreate/>}/>

                    <Route path="data" element={ <DataMain/> } />

                    <Route path="visualization" element={ <VisualMain/> } />
                </Route>

                {/* 로그인 전 페이지 - 로그인, 회원 가입 */}
                <Route path="/login" element={ <PublicRoute> <Login/> </PublicRoute>} />
                <Route path="/signup" element={ <PublicRoute> <SignUp/> </PublicRoute>} />

                {/* 페이지 URL 오류 */}
                <Route path="*" element={ <NotFound/> } />

            </Routes>
        </BrowserRouter>
    );
}

export default App;
