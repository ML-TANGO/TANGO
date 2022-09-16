import React from "react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PlusLg } from "react-bootstrap-icons";

import * as Request from "../../../service/restProjectApi";

import Kebab from "../../../components/Kebab/Kebab";
import Progress from "../../../components/Progress/Progress";
import * as Popup from "../../../components/popup/popup";

import '../../../CSS/project_management.css'

import { isToken, isUser } from '../../../Router/isLogin';

import data_th_1 from "../../../images/thumbnail/data_th_1.PNG";   // 칫솔
import data_th_2 from "../../../images/thumbnail/data_th_2.PNG";   // 용접 파이프
import data_th_3 from "../../../images/thumbnail/data_th_3.PNG";   // 실생활
import data_th_4 from "../../../images/thumbnail/data_th_4.PNG";   // 폐결핵 판독

function ProjectMain()
{
    /* 페이지 이동을 위한 useNavigate 객체 추가 */
    const navigate = useNavigate();

    /* 서버에서 받아온 프로젝트 리스트 정보 */
    const [projectList, setProjectList] = useState([]);

    const [projectName, setProjectName] = useState("");                   // 프로젝트 이름
    const [projectDescription, setProjectDescription] = useState("");     // 프로젝트 설명

    /* 프로젝트 이름 수정을 위한 state 변수 : 프로젝트 아이디, 이전 이름, 수정 이름*/
    const [modifyProjectId, setModifyProjectId] = useState("");
    const [oriProjectName, setOriProjectName] = useState("");
    const [newProjectName, setNewProjectName] = useState("");

    const type_backgroundColor = {
        'detection' : '#ED7D31',
        'classification' : '#70AD47',
        'segmentation' : '#0070C0'
    }

    /* 페이지 로드 완료시 호출 이벤트 */
    useEffect( () => {

        // 20220309 jpchoi
        if(  !!isToken() === false || !!isUser() === false )
        {
            /* 로그인 상태가 아닌 경우  */
            window.location.replace("/");
        }
        else
        {
            getProject();
        }

    }, []);

    /* 프로젝트 리스트 불러오기 */
    const getProject = () => {
        Request.requestProjectList().then(result =>
        {
            setProjectList(result.data);
        })
        .catch(error =>
        {
            console.log('project list get error')
        });
    };

    /* 프로젝트 삭제 */
    const delProject = (id, name) => {
        Request.requestProjectDelete(id).then(result =>
        {
            getProject();
        })
        .catch(error =>
        {
            console.log('project delete error')
        });
    };

    /* 프로젝트 생성 버튼 클릭 */
    const project_createButtonClick = () =>
    {
        /* 팝업 보이기 */
        document.getElementById('create_project_popup').style.display = 'block';
    }

    /* 프로젝트 생성 */
    const createProject = (name, description) => {

        var name_check = name.trim();

        if (name_check.length > 0)
        {
            const data = {
                name:name,
                description:description
            }
            Request.requestProjectCreate(data).then(result =>
            {
                /* 프로젝트 이름이 중복되는 경우 */
                if(result.data['result'] === false)
                {
                    alert('프로젝트 이름이 중복됩니다.')
                }
                else
                {
                    /* 프로젝트 생성 팝업의 이름 입련란 초기화 */
                    document.getElementById('input_project_name').value = ''

                    const new_project_id = result.data['id']
                    const new_project_name = result.data['name']
                    const new_project_description = result.data['description']

                    /* DB에 프로젝트 생성이 완료된 경우 페이지 이동 */
                    /* 전달 파라미터 = state : 프로젝트 이름, 프로젝트 아이디 */
                    navigate('create', {state : { id: new_project_id, name: new_project_name, description: new_project_description }} )
                }
            })
            .catch(error =>
            {
                console.log('project create error')
            });
        }
        else
        {
            setProjectName('')
            alert('프로젝트 이름을 확인해주세요.')
        }

    };

    /* 생성 팝업 - 생성 버튼 클릭 */
    const popup_Create_ButtonClick = () =>
    {
        /* 프로젝트 생성 함수 호출 */
        /* projectName : 사용자가 입력한 프로젝트명 */
        createProject(projectName, projectDescription)
    }

    /* 생성 팝업 - 취소 버튼 클릭 */
    const popup_Cancel_ButtonClick = () =>
    {
        document.getElementById('input_project_name').value = ''

        /* 팝업 숨김 */
        document.getElementById('create_project_popup').style.display = 'none';
    }

    /* 프로젝트 이름 수정 */
    const modifyProject = (id, name) =>
    {
        setModifyProjectId(id)
        setOriProjectName(name)
        setNewProjectName(name)
        /* 이름 수정 팝업 보이기 */
        document.getElementById('project_name_modify_popup').style.display = 'block';
    };

    /* 수정 팝업 - 수정 버튼 클릭 */
    const modify_popup_Apply_ButtonClick = () =>
    {
        if (oriProjectName === newProjectName)
        {
            alert('동일한 프로젝트 이름입니다.')
        }
        else
        {
            var name_check = newProjectName.trim();

            if (name_check.length > 0)
            {
                const param = {
                    'id':modifyProjectId,
                    'name':newProjectName
                }
                Request.requestProjectRename(param).then(result =>
                {
                    /* 프로젝트 이름이 중복되는 경우 */
                    if(result.data['result'] === false)
                    {
                        alert('프로젝트 이름이 중복됩니다.')
                    }
                    else
                    {
                        document.getElementById('project_name_modify_popup').style.display = 'none';
                        getProject();
                    }
                })
                .catch(error =>
                {
                    alert('project rename error')
                });
            }
            else
            {
                setProjectName('')
                alert('프로젝트 이름을 확인해주세요.')
            }

        }
    }

    /* 수정 팝업 - 취소 버튼 클릭 */
    const modify_popup_Cancel_ButtonClick = () =>
    {
        setModifyProjectId('')
        setOriProjectName('')
        setNewProjectName('')

        document.getElementById('input_modify_name').value = ''
        /* 이름 수정 팝업 숨김 */
        document.getElementById('project_name_modify_popup').style.display = 'none';
    }

    /* 아이템 박스 배경색 */
    const backgroundColorChange = (value) =>
    {
        return type_backgroundColor[value]
    }

    /* 아이템 박스 이미지 */
    const itemBackgroundImage = (value) =>
    {
        if (value.indexOf('COCO') !== -1)
        {
            return "url('" + data_th_3 + "')";
        }
        else if (value.indexOf('칫솔') !== -1)
        {
            return "url('" + data_th_1 + "')";
        }
        else if (value.indexOf('파이프') !== -1)
        {
            return "url('" + data_th_2 + "')";
        }
        else if (value.indexOf('폐') !== -1)
        {
            return "url('" + data_th_4 + "')";
        }
        else
        {
            return "";
        }
    }


    return (
        <>
        {/* 프로젝트 이름 생성 팝업 */}
        <Popup.ProjectCreatePopup
            projectName={projectName}
            setProjectName={setProjectName}
            projectDescription={projectDescription}
            setProjectDescription={setProjectDescription}
            cancel_ButtonClick={() => popup_Cancel_ButtonClick()}
            create_ButtonClick={popup_Create_ButtonClick}/>

        {/* 프로젝트 이름 수정 팝업 */}
        <Popup.ProjectRenamePopup
            getProject={getProject}
            newProjectName={newProjectName}
            setNewProjectName={setNewProjectName}
            cancel_ButtonClick={() => modify_popup_Cancel_ButtonClick()}
            apply_ButtonClick={modify_popup_Apply_ButtonClick}/>


        {/* 프로젝트 메인 페이지 */}
        <div className='manage_list_container'>

            {/* 프로젝트 메인 페이지 - 헤더 */}
            <div className='manage_header' style={{width:'100%'}}>
                {/* 이동경로 레이아웃 */}
                <div className='path'></div>

                <div className='title'>
                    <div className='title_left'>
                        Project Management
                    </div>

                    <div className='title_right'>
                        <div onClick={ () => project_createButtonClick() }>
                            <button type='button' id='create_button'> <PlusLg size={19} color="#C3c8d3" />&nbsp; Create Project</button>
                        </div>
                    </div>
                </div>
            </div>

            {/* 프로젝트 리스트 */}
            <div className='project_manage_content' style={{display:'block'}}>

                { projectList.length > 0 ?
                    <>
                    <div className='project_manage_list' style={{height:'auto', width:'100%'}}>
                        {/* 프로젝트 리스트가 1개 이상인 경우 */}
                        {projectList.map((menu, index) => {
                            return (
                                <Link className='item_box' key={index} style={{backgroundColor:backgroundColorChange(menu.type)}} to={'create'} state={{ name: menu.project_name, id: menu.id, description: menu.project_description }} >
                                    <Kebab index={index} page={'project'} itemID={menu.id} itemName={menu.project_name} deleteItem={delProject} modifyItem={modifyProject} deleteAlter={"프로젝트를"} />
                                    <div className='item_title'>{menu.project_name}</div>

                                    <div id='item_image' className='item_image' style={{backgroundImage:itemBackgroundImage(menu.dataset_path), backgroundColor:'white', borderRadius:'5px'}}>
                                    { menu.dataset_path === '' &&
                                        <div className="image_text">Please complete the project creation</div>
                                    }
                                    </div>
                                    <div className='item_content' style={{backgroundColor:'white', borderRadius:'5px'}}>{menu.type}</div>
                                </Link>
                            )
                        })}
                    </div>
                    </>
                    :
                    <>
                    {/* 프로젝트 리스트가 없는 경우 */}
                    <div style={{height:'100%', width:'100%',
                        fontSize:"50px", fontWeight:'700', textAlign:'center',
                        display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                        Please Create Project!
                    </div>
                    </>
                }
            </div>
        </div>
        </>
    );
}

export default ProjectMain;