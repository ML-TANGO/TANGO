import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar/LayerToggle";
import axios from 'axios';



const Shortcut = (props) => {
  var text_value = ''  // 변수 선언
  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

  for (var i=0; i<parmArr.length; i++){
    var param = parmArr[i].replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
    var eachParam = param.split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

    switch(i){  // 파라미터별로 해당 객체 값 설정
        case 0:
          text_value = eachParam[1];
          break;
    }
  }

  const [text, setText] = React.useState(text_value);
  const { open, save, close, header } = props;

  const bfsave=(event)=>{
    //console.log('props.params', props.params)
    var send_message = "'dim': ".concat(text)

    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
      order: String(props.layer),
      layer: "Shortcut",
      parameters: send_message
    }).then(function(response){
      console.log(response)
    }).catch(err=>console.log(err));
    // node update하기 ********************

    props.setState("");
    save();
  };


  return (
    <div className={open ? 'openModal modal' : 'modal'}>
      {open ? (
        <section>
          <header>
            {header}
          </header>
          <main>
          <React.Fragment>

          <div>
              <li>
                  <label htmlFor="text">dim:</label>
                  <EditText name="dim" type="number" style={{width: '50px'}} value={text}
                    onChange={setText} inline/>
              </li>
          </div>

          </React.Fragment>
          </main>
          <div className="btnDiv">
            <button className="close" onClick={() => {setText('1')}}>
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

export default Shortcut;
