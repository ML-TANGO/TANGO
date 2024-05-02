import React, { useState } from 'react';
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar/LayerToggle";


import axios from 'axios';

const IDetect = (props) => {

  var text_value = ''  // 변수 선언
  var text2_value = ''
  var text3_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

   for (var i=0; i<parmArr.length; i++){
        var param = String(parmArr[i]).replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
        var eachParam = String(param).split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

        switch(i){  // 파라미터별로 해당 객체 값 설정
            case 0:  // 'nc'
                text_value = String(eachParam[1]);
                break;
            case 1:  // 'anchors'
                text2_value = String(eachParam[1]);
                break;
            case 2:  // 'ch'
                text3_value = String(eachParam[1]);
                break;
        }
    }

  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);
  const [text3, setText3] = React.useState(text3_value);

  // 수정된 부분
  const { open, save, close, header } = props;

  const bfsave=(event)=>{

    var send_message = "'nc': ".concat(text)
        .concat(" \n 'anchors': ").concat(text2)
        .concat(" \n 'ch': ").concat(text3)

    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "IDetect",
        parameters: send_message
    }).then(function(response){
        console.log(response)
    }).catch(err=>console.log(err));
    // node update하기 ********************

    props.setState("");

    save();
  };

  // params 불러오기 *****************
  return (
    <div className={open ? 'openModal modal' : 'modal'}>
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
                      <label htmlFor="text">number of classes:</label>
                      <EditText name="nc" type="number" style={{width: '50px'}} value={text} onChange={setText} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">anchors:</label>
                      <EditText name="anchors" type="number" style={{width: '50px'}} value={text2}
                onChange={setText2} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">channels:</label>
                      <EditText name="ch" type="number" style={{width: '40px'}} value={text3}
                onChange={setText3} inline/>
                  </li>
              </div>



          </React.Fragment>
          </main>
          <div className="btnDiv">
              <button className="close" onClick={() => {
              setText('80')
              setText2('')
              setText3('')
              }} >
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

export default IDetect;
