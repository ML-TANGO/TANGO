import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar/LayerToggle";
import axios from 'axios';



const ReOrg = (props) => {
  console.log('String(props)', String(props));

  const { open, save, close, header } = props;

  // const bfsave=(event)=>{
  //   var send_message = ""
  //   // node update하기 ********************
  //   axios.put("/api/node/".concat(String(props.layer).concat('/')),{
  //     order: String(props.layer),
  //     layer: "ReOrg",
  //     parameters: send_message
  //   }).then(function(response){
  //     console.log(response)
  //   }).catch(err=>console.log(err));
  //   // node update하기 ********************

  //   props.setState("");
  //   save();
  // };


  return (
    <div className={open ? 'openModal modal' : 'modal'}>
      {open ? (
        <section>
          <header>
            {header}
          </header>
          <main>
          <React.Fragment>
            <div></div>
          </React.Fragment>
          </main>
          <div className="btnDiv">
            <button className="close" onClick={() => {}}>
              default
            </button>
            <button className="save" onClick={save}>
              save
            </button>
          </div>
        </section>
      ) : null}
    </div>
  );