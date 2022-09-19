import React, { useEffectm, useState } from "react";
import { Outlet, Routes, Route } from "react-router-dom";

/* 레이아웃 컴포넌트 - 사이드 바 & 헤더 */
import Sidebar from "../../components/sidebar/sidebar";
import Header from "../../components/header/header";
import ProjectCreate from "../../templates/main/project/projectCreate";

import '../../CSS/main.css';
import '../../CSS/sidebar.css';
import '../../CSS/tooltip.css';

import { Link } from "react-router-dom";
import { List } from "react-bootstrap-icons";

function Main()
{
    const [sideHidden, setSideHidden] = useState(false);

    const menu_buttonClick = () =>
    {
        if(sideHidden == false)
        {
            setSideHidden(true);
            document.getElementById('sidebar').style.width = '60px';

        }
        else
        {
            setSideHidden(false);
            document.getElementById('sidebar').style.width = 'auto';

        }
    }

    /* 메인페이지 레이아웃 분할 */
    return (
        <div className="container" style={{backgroundColor:'white'}}>

            <div className="header">
                <Header/>
            </div>

            {/* 사이드 바 */}
            <div id='sidebar' className="sidebar" style={{backgroundColor:'#303030',  height:'100%', width:'100%'}}>
                {/* 사이드 바 컴포넌트 */}

                <div className="sidebar_item">
                    <div className='burger_item'>
                        <List color="#C2C8D3" size="30px" onClick={ () =>menu_buttonClick()}/>
                    </div>
                </div>

                { sideHidden === false
                ?
                    <Sidebar style={{width:'300px'}}/>
                :
                    <div className="short_sidebar_item" style={{marginTop:'20px'}}>
                        <ul>
                            <li className="short">
                                <Link to="/">
                                    <div className='short_item'>
                                        <span className="short_sidebar_item tooltip" id='home_icon'><span className="tooltiptext" style={{width:'100px'}}>Home</span></span>
                                    </div>
                                </Link>
                            </li>

                            <li className="short" style={{marginTop:'20px'}}>
                                <Link to="/project">
                                    <div className='short_item'>
                                        <span className="short_sidebar_item tooltip" id='project_icon'><span className="tooltiptext" style={{width:'200px'}}>Project Management</span></span>
                                    </div>
                                </Link>
                            </li>

                            <li className="short">
                                <Link to="/target">
                                    <div className='short_item'>
                                        <span className="short_sidebar_item tooltip" id='target_icon'><span className="tooltiptext" style={{width:'200px'}}>Target Management</span></span>
                                    </div>
                                </Link>
                            </li>

                            <li className="short">
                                <Link to="/data">
                                    <div className='short_item'>
                                        <span className="short_sidebar_item tooltip" id='data_icon'><span className="tooltiptext" style={{width:'200px'}}>Data Management</span></span>
                                    </div>
                                </Link>
                            </li>

                            <li className="short">
                                <Link to="/visualization">
                                    <div className='short_item'>
                                        <span className="short_sidebar_item tooltip" id='visual_icon'><span className="tooltiptext" style={{width:'150px'}}>Visualization</span></span>
                                    </div>
                                </Link>
                            </li>
                        </ul>
                    </div>
                }
            </div>

            {/* 컨텐츠 */}
            <div className="content" style={{ width:'100%', height:'100%', padding:'0px 25px 0px 25px'}} >

                {/* 홈, 프로젝트, 데이터 관리 페이지 이동 Route */}
                <Outlet/>
            </div>

            {/* 바닥 */}
            <div className="footer" style={{height:'100%', width:'100%'}}></div>

        </div>
    );
}

export default Main;
