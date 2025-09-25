import React, { useState } from "react";
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea } from "react-edit-text";
import "react-edit-text/dist/index.css";
import axios from 'axios';

const Bottleneck = (props) => {

  var text_value = ''  // 변수 선언
  var text2_value = ''
  var text3_value = ''
  var radio1_value = ''
  var text5_value = ''
  var text6_value = ''
  var text7_value = ''
  var text8_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

   for (var i=0; i<parmArr.length; i++){
        var param = String(parmArr[i]).replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
        console.log('param', param)
        var eachParam = String(param).split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

        console.log('String(props.params)', String(props.params));

        switch(i){  // 파라미터별로 해당 객체 값 설정
            case 0:
                text_value = String(eachParam[1]);
                break;
            case 1:
                text2_value = String(eachParam[1]);
                break;
            case 2:  // 'kernel_size': (3, 3) 이므로, 괄호 안에서 3과 3을 따로 분리해주어야함
                text3_value = String(eachParam[1]);
                break;

            case 3:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                //console.log(typeof(eachParam[1]))
//                if (String(eachParam[1]) === 'None'){
                radio1_value = String(eachParam[1]);
//                }
//
//                else{
//                    text4_value = Number(eachParam[1]);
//
//
//                }

                break;
            case 4:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                text5_value = String(eachParam[1]);
                break;
            case 5:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                text6_value = String(eachParam[1]);
                break;
            case 6:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                text7_value = String(eachParam[1]);
                break;
            case 7:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                text8_value = String(eachParam[1]);
                break;


        }
    }
  //라디오 버튼 값 조회
  const handleClickRadioButton1 = (e) => {
    console.log(e.target.value)
    setRadio1(e.target.value)
  }
  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);
  const [text3, setText3] = React.useState(text3_value);
  const [radio1, setRadio1] = React.useState(radio1_value);
  const [text5, setText5] = React.useState(text5_value);
  const [text6, setText6] = React.useState(text6_value);
  const [text7, setText7] = React.useState(text7_value);
  const [text8, setText8] = React.useState(text8_value);


  // 수정된 부분
  const { open, save, close, header } = props;

  const bfsave=(event)=>{

    var send_message = "'inplanes': ".concat(text)
        .concat(" \n 'planes': ").concat(text2)
        .concat(" \n 'stride': ").concat(text3)
        .concat(" \n 'downsample': ").concat(radio1)
        .concat(" \n 'groups': ").concat(text5)
        .concat(" \n 'base_width': ").concat(text6)
        .concat(" \n 'dilation': ").concat(text7)
        .concat(" \n 'norm_layer': ").concat(text8)
      console.log(send_message);
      // node update하기 ********************
      axios.put("/api/node/".concat(String(props.layer).concat('/')),{
          order: String(props.layer),
          layer: "Bottleneck",
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
            Bottleneck
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
                  <label htmlFor="text">inplanes:</label>
                <EditText
                  name="inplanes"
                  type="number"
                  style={{ width: "40px" }}
                  value={text}
                  onChange={setText}
                  inline
                />
                </li>
                <li>
                  <label htmlFor="text">planes:</label>
                <EditText
                  name="planes"
                  type="number"
                  style={{ width: "40px" }}
                  value={text2}
                  onChange={setText2}
                  inline
                />
                </li>
                <li>
                  <label htmlFor="text">stride:</label>
                <EditText
                  name="stride"
                  type="number"
                  style={{ width: "40px" }}
                  value={text3}
                  onChange={setText3}
                  inline
                />
                </li>
                <li>
                  <label htmlFor="text">downsample:</label>
                  <label> <input type="radio" name="radio1" value="True" onChange={handleClickRadioButton1} checked={radio1.includes("T")===true ? true : false}/>True </label>
                  <label> <input type="radio" name="radio1" value="False" onChange={handleClickRadioButton1} checked={radio1.includes("F")===true ? true : false}/>False </label>
                </li>
                <li>
                  <label htmlFor="text">groups:</label>
                <EditText
                  name="groups"
                  type="number"
                  style={{ width: "40px" }}
                  value={text5}
                  onChange={setText5}
                  inline
                />
                </li>
                <li>
                  <label htmlFor="text">base_width:</label>
                <EditText
                  name="base_width"
                  type="number"
                  style={{ width: "40px" }}
                  value={text6}
                  onChange={setText6}
                  inline
                />
                </li>
                <li>
                  <label htmlFor="text">dilation:</label>
                <EditText
                  name="dilation"
                  type="number"
                  style={{ width: "40px" }}
                  value={text7}
                  onChange={setText7}
                  inline
                />
                </li>
                <li>
                  <label htmlFor="text">norm_layer:</label>
                <EditText
                  name="norm_layer"
                  type="text"
                  style={{ width: "40px" }}
                  value={text8}
                  onChange={setText8}
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
                setText("1");
                //setRadio1('False');
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

export default Bottleneck;