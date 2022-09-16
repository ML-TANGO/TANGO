import React, { useState } from "react";
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import {EditText} from "react-edit-text";
import "react-edit-text/dist/index.css";
import axios from 'axios';

const ConstantPad2d = (props) => {

  var text_value = ''  // 변수 선언
  var text2_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

  for (var i=0; i<parmArr.length; i++){
        var param = String(parmArr[i]).replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
        var eachParam = String(param).split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

        switch(i){  // 파라미터별로 해당 객체 값 설정
            case 0:  // 'kernel_size': (3, 3) 이므로, 괄호 안에서 3과 3을 따로 분리해주어야함
                text_value = String(eachParam[1]);
                break;
            case 1:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                text2_value= String(eachParam[1]);
                break;
        }
    }


  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);

  const { open, save, header } = props;

  const bfsave=(event)=>{

    //console.log('radio1', radio1);
  
    console.log('props.params', props.params)
  
      var send_message = "'padding': ".concat(text).concat(" \n 'value': ").concat(text2)
  
      console.log(send_message);
      // node update하기 ********************
      axios.put("/api/node/".concat(String(props.layer).concat('/')),{
          order: String(props.layer),
          layer: "ConstantPad2d",
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
                setText("2");
                setText2("3.5");
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
                  name="padding"
                  type="number"
                  style={{ width: "40px" }}
                  value={text}
                  onChange={setText}
                  inline
                />
              </div>
              <div>
                <label htmlFor="text">value:</label>
                <EditText
                  name="value"
                  type="text"
                  style={{ width: "40px" }}
                  value={text2}
                  onChange={setText2}
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

export default ConstantPad2d;