import {NavLink} from "react-router-dom";
import layer_icon from "../../img/layer_icon.png";
import info_icon from "../../img/info_icon.png";
import abstract_icon from "../../img/abstract_icon.png";
import code_icon from "../../img/code_icon.png";
import layer_click_icon from "../../img/layer_click_icon.png";
import info_click_icon from "../../img/info_click_icon.png";
import {useState} from "react";
const Tab = ({tabOnClick}) => {

    const tabList = [
        {
            path: "/",
            image: layer_icon,
            alt: "layer icon"
        },
        //{
        //    path: "/info",
        //    image: info_icon,
       //     alt: "info icon"
       // },
        // {
        //     path: "/abstract",
        //     image: abstract_icon,
        //     alt: "abstract_icon"
        // },
        // {
        //     path: "/code",
        //     image: code_icon,
        //     alt: "code icon"
        ];

    return(
        <div className="Tab">
            {
                tabList.map((tab) => (
                    <NavLink to={tab.path} className={({isActive}) => isActive ? "selected" : ""}>
                        <img onClick={(e)=>{tabOnClick(e.target.alt)}} className="tab_image" src={tab.image} alt={tab.alt}/>
                    </NavLink>
                ))
            }
        </div>
    )
}

export default Tab;