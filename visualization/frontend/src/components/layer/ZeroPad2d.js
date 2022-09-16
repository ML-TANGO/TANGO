import React, { useState } from "react";
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea } from "react-edit-text";
import "react-edit-text/dist/index.css";
import axios from 'axios';

const ZeroPad2d = (props) => {

  var text_value = ''  // 변수 선언
  var param = String(props.params).replace('"','');
  var eachParam = String(param).split(': ');
  text_value = eachParam[1];

  const [text, setText] = React.useState(text_value);

  const { open, save, header } = props;

  const bfsave=(event)=>{
    console.log('props.params', props.params)
  
      var send_message = "'padding': ".concat(text)
  
      console.log(send_message);
      // node update하기 ********************
      axios.put("/api/node/".concat(String(props.layer).concat('/')),{
          order: String(props.layer),
          layer: "ZeroPad2d",
          parameters: send_message
      }).then(function(response){
          console.log(response)
      }).catch(err=>console.log(err));
      // node update하기 ********************
  
  //    console.log(text8, "text7");
  //    console.log(text9, "text7");
  
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

            <button
              className="close"
              onClick={() => {
                setText("1");
              }}
            >
              default
            </button>
            <button className="save" onClick={bfsave}>
              save
            </button>
            {/* <button className="close" onClick={close}>
              &times;
            </button> */}
          </header>
          <main>
            <React.Fragment>
              <div>
                <label htmlFor="text">padding:</label>
                <EditText
                  name="ZeroPad2d"
                  type="number"
                  style={{ width: "40px" }}
                  value={text}
                  onChange={setText}
                  inline
                />
              </div>
            </React.Fragment>
          </main>
          {/* <main>{val}</main> */}
          {/* <footer>
            
          </footer> */}
        </section>
      ) : null}
    </div>
  );
};

export default ZeroPad2d;