import React, { useState } from 'react';
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar/LayerToggle";


import axios from 'axios';

const SPPCSPC = (props) => {

  var text_value = ''   // in_channels
  var text2_value = ''  // out_channels
  var text3_value = ''  // n
  var radio1_value = '' // shortcut (T or F)
  var text4_value = ''  // groups
  var text5_value = ''  // expansion
  var text6_value = ''  // kernels - small
  var text7_value = ''  // kernels - medium
  var text8_value = ''  // kernels - large

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
            case 2:
                text3_value = String(eachParam[1]);
                break;
            case 3:
                radio1_value = String(eachParam[1]);
                break;
            case 4:
                text4_value = String(eachParam[1]);
                break;
            case 5:
                text5_value = String(eachParam[1]);
                break;
            case 6:
                var kernelArray = String(eachParam[1]).split(', ');
                text6_value = String(kernelArray[0]).replace("(", "");
                text7_value = String(kernelArray[1])
                text8_value = String(kernelArray[2]).replace(")", "");
                break;
        }
  }

  const handleClickRadioButton1 = (e) => {
    setRadio1(e.target.value)
  }
  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);
  const [text3, setText3] = React.useState(text3_value);
  const [text4, setText4] = React.useState(text4_value);
  const [text5, setText5] = React.useState(text5_value);
  const [text6, setText6] = React.useState(text6_value);
  const [text7, setText7] = React.useState(text7_value);
  const [text8, setText8] = React.useState(text8_value);
  const [radio1, setRadio1] = React.useState(radio1_value);

  // 수정된 부분
  const { open, save, close, header } = props;

  const bfsave=(event)=>{

    var send_message = "'in_channels': ".concat(text)
        .concat(" \n 'out_channels': ").concat(text2)
        .concat(" \n 'n': ").concat(text3)
        .concat(" \n 'shortcut': ").concat(radio1)
        .concat(" \n 'groups': ").concat(text4)
        .concat(" \n 'expansion': ").concat(text5)
        .concat(" \n 'kernels': (").concat(text6).concat(', ').concat(text7).concat(', ').concat(text8).concat(")")

    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "SPPCSPC",
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
                      <EditText
                        name="in_channels"
                        type="number"
                        style={{width: '50px'}}
                        value={text}
                        onChange={setText} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">out_channels:</label>
                      <EditText
                        name="out_channels"
                        type="number"
                        style={{width: '50px'}}
                        value={text2}
                        onChange={setText2} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">n:</label>
                      <EditText
                        name="n"
                        type="number"
                        style={{width: '50px'}}
                        value={text3}
                        onChange={setText3} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">shortcut: </label>
                      <label>
                        <input type="radio"
                          name="radio1"
                          value="True"
                          onChange={handleClickRadioButton1}
                          checked={radio1.includes("T")===true ? true : false}/>
                            True
                      </label>
                      <label>
                        <input type="radio"
                          name="radio1"
                          value="False"
                          onChange={handleClickRadioButton1}
                          checked={radio1.includes("F")===true ? true : false}/>
                            False
                      </label>
                  </li>
                  <li>
                      <label htmlFor="text">groups:</label>
                      <EditText
                        name="groups"
                        type="number"
                        style={{width: '50px'}}
                        value={text4}
                        onChange={setText4} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">expansion:</label>
                      <EditText
                        name="expansion"
                        type="number"
                        style={{width: '50px'}}
                        value={text5}
                        onChange={setText5} inline/>
                  </li>
                  <li>
                      <label htmlFor="text">kernels:</label>
                      <EditText
                        name="kernels1"
                        type="number"
                        style={{width: '40px'}}
                        value={text6}
                        onChange={setText6} inline/>
                      <label htmlFor="text">, </label>
                      <EditText
                        name="kernels2"
                        type="number"
                        style={{width: '40px'}}
                        value={text7}
                        onChange={setText7} inline/>
                      <label htmlFor="text">, </label>
                      <EditText
                        name="kernels3"
                        type="number"
                        style={{width: '40px'}}
                        value={text8}
                        onChange={setText8} inline/>
                  </li>
              </div>



          </React.Fragment>
          </main>
          <div className="btnDiv">
              <button className="close" onClick={() => {
                setText('64')
                setText2('64')
                setText3('1')
                setRadio1('False')
                setText4('1')
                setText5('0.5')
                setText6('5')
                setText7('9')
                setText8('13')
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

export default SPPCSPC;
