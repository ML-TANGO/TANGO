import React, { useState } from "react";
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea } from "react-edit-text";
import "react-edit-text/dist/index.css";
import Sidebar from "../sidebar/LayerToggle";
import axios from 'axios';

const Softmax = (props) => {

  var text_value = ''  // 변수 선언
  var param = String(props.params).replace('"','');
  var eachParam = String(param).split(': ');
  text_value = eachParam[1];

  const [text, setText] = React.useState(text_value);

  const { open, save, header } = props;

  const bfsave=(event)=>{
    console.log('props.params', props.params)

      var send_message = "'dim': ".concat(text)

      console.log(send_message);
      // node update하기 ********************
      axios.put("/api/node/".concat(String(props.layer).concat('/')),{
          order: String(props.layer),
          layer: "Softmax",
          parameters: send_message
      }).then(function(response){
          console.log(response)
      }).catch(err=>console.log(err));
      // node update하기 ********************

  //    console.log(text8, "text7");
  //    console.log(text9, "text7");
props.setState("");
      save();
    };

  return (
    <div className={open ? "openModal modal" : "modal"}>
      {open ? (
        <section>
          <header>
            {header}
            {/* { <button className="save" onClick={close}>
              Save
            </button> } */}


            {/* <button className="close" onClick={close}>
              &times;
            </button> */}
          </header>
          <main>
            <React.Fragment>
              <div>
                <li>
                  <label htmlFor="text">dim:</label>
                  <EditText
                  name="dim"
                  type="number"
                  style={{ width: "40px" }}
                  value={text}
                  onChange={setText}
                  inline
                />
                </li>
              </div>
            </React.Fragment>
          </main>
          <div className="btnDiv">
          <button
              className="close"
              onClick={() => {
                setText("0");
              }}
            >
              default
            </button>
            <button className="save" onClick={bfsave}>
              save
            </button>
            </div>
        </section>
      ) : null}
    </div>
  );
};

export default Softmax;