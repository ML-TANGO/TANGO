import React, { useState } from 'react';
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar/LayerToggle";


import axios from 'axios';

const Conv = (props) => {

  var text_value = ''  // 변수 선언
  var text2_value = ''
  var text3_value = ''
  var text4_value = ''
  var text5_value = ''
  var text6_value = ''
  var text7_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

   for (var i=0; i<parmArr.length; i++){
        var param = String(parmArr[i]).replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
        var eachParam = String(param).split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

        switch(i){  // 파라미터별로 해당 객체 값 설정
            case 0:
                text_value = String(eachParam[1]);
                break;
            case 1:
                text2_value = String(eachParam[1]);
                break;
            case 2:  // 'k'
                text3_value = String(eachParam[1]);
                break;
            case 3:  // 's'
                text4_value = String(eachParam[1])
                break;
            case 4:  // 'p'
                text5_value = String(eachParam[1]);
                break;
            case 5:  // 'g'
                text6_value = String(eachParam[1]);
                break;
            case 6: // 'act'
                text7_value = String(eachParam[1]);
                break;
        }
    }

  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);
  const [text3, setText3] = React.useState(text3_value);
  const [text4, setText4] = React.useState(text4_value);
  const [text5, setText5] = React.useState(text5_value);
  const [text6, setText6] = React.useState(text6_value);
  const [text7, setText7] = React.useState(text7_value);

  // 수정된 부분
  const { open, save, close, header } = props;

  const bfsave=(event)=>{

    var send_message = "'in_channels': ".concat(text)
        .concat(" \n 'out_channels': ").concat(text2)
        .concat(" \n 'kernel_size': ").concat(text3)
        .concat(" \n 'stride': ").concat(text4)
        .concat(" \n 'padding': ").concat(text5)
        .concat(" \n 'groups': ").concat(text6)
        .concat(" \n 'act': ").concat(text7)

    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "Conv",
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
                      <label htmlFor="text">in_channels:</label>
                      <EditText name="in_channels" type="number" style={{width: '50px'}} value={text} onChange={setText} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">out_channels:</label>
                      <EditText name="out_channels" type="number" style={{width: '50px'}} value={text2}
                onChange={setText2} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">kernel_size:</label>
                      <EditText name="kernel_size" type="number" style={{width: '40px'}} value={text3}
                onChange={setText3} inline/>
                      <label htmlFor="text"> X </label>
                      <EditText name="kernel_size2" type="number" style={{width: '40px'}} value={text3}
                onChange={setText3} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">stride:</label>
                      <EditText name="stride" type="number" style={{width: '40px'}} value={text4}
                onChange={setText4} inline/>
                      <label htmlFor="text"> X </label>
                      <EditText name="stride2" type="number" style={{width: '40px'}} value={text4}
                onChange={setText4} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">padding:</label>
                      <EditText name="padding" type="text" style={{width: '40px'}} value={text5}
                onChange={setText5} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">group:</label>
                      <EditText name="group" type="number" style={{width: '40px'}} value={text6}
                onChange={setText6} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">activation:</label>
                      <EditText name="act" type="text" style={{width: '50px'}} value={text7}
                onChange={setText7} inline/>
                  </li>
              </div>



          </React.Fragment>
          </main>
          <div className="btnDiv">
              <button className="close" onClick={() => {
              setText('64')
              setText2('64')
              setText3('1')
              setText4('2')
              setText5('None')
              setText6('1')
              setText7('True')
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

export default Conv;
