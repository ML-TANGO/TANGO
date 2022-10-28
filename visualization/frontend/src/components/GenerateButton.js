import React from 'react';
import axios from 'axios';
import "../styles.css";
import styled from "styled-components";

function GenerateButton(props){
    const onShow=(event)=>{
        event.preventDefault();
        var data = props.elements;
        data = (Object.values((Object.entries(data))));

        axios.post("/api/pth/")
        .then(function(response){
        console.log(response)
        })
        .catch(e => console.log(e))
    };

    return(
        <div>
            <button class="btn_fin" onClick={onShow}> 생성 </button>
        </div>
    )
}

export default GenerateButton;
