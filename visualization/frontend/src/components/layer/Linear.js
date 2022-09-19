import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar";

import axios from 'axios';


const Linear = (props) => {

  console.log('String(props.params)', String(props.params));

  var text_value = ''  // 변수 선언
  var text2_value = ''
  var radio1_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

  for (var i=0; i<parmArr.length; i++){
    var param = parmArr[i].replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
    var eachParam = param.split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

    switch(i){  // 파라미터별로 해당 객체 값 설정
        case 0:
            text_value = eachParam[1];
            break;
        case 1:
            text2_value = eachParam[1];
            break;
        case 2:
            radio1_value = eachParam[1];
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
  const [radio1, setRadio1] = React.useState(radio1_value);


  // 수정된 부분
  const { open, save, close, header } = props;

//  let isCheckedrt
//  let isCheckedct
//  let isCheckedrf
//  let isCheckedcf
//

  const bfsave=(event)=>{

  //console.log('radio1', radio1);

  //console.log('props.params', props.params)

    var send_message = "'in_features': ".concat(text)
        .concat(" \n 'out_features': ").concat(text2)
        .concat(" \n 'bias': ").concat(radio1)


    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "Linear",
        parameters: send_message
    }).then(function(response){
        console.log(response)
    }).catch(err=>console.log(err));
    // node update하기 ********************

//    console.log(text8, "text7");
//    console.log(text9, "text7");

    save();
  };
  //params 불러오기 ******;***********
  //console.log("max실행");
  //console.log("params", props.params);
  //console.log("params", String(props.params).charAt(20));
//  console.log("params", String(props.params).CharAt(25));
//  console.log("params", String(props.params).CharAt(17));
//  console.log("params", String(props.params).CharAt(17));

//  const bfdelete=(event)=>{

//     axios.delete("/api/node/".concat(String(props.layer).concat('/')))
//   .then(function (response) {
//         console.log(response)
//   })
//   .catch(function (error) {
//     // handle error
//   })
//   .then(function () {
//     // always executed
//   });
//   close();
//   };

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

            <button className="close" onClick={() => {
              setText('1')
              setText2('1')
              setRadio1('False')
              }
  } >
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


          <div><label htmlFor="text">in_features:</label>
          <EditText name="in_features" type="number" style={{width: '50px'}} value={text}
            onChange={setText} inline/></div>
          <div><label htmlFor="text">out_features:</label>
          <EditText name="out_features" type="number" style={{width: '50px'}} value={text2}
            onChange={setText2} inline/></div>
          <div><label htmlFor="text">bias:</label>
          <label> <input type="radio" name="radio1" value="True" onChange={handleClickRadioButton1} checked={radio1.includes("T")===true ? true : false}/>True </label>
          <label> <input type="radio" name="radio1" value="False" onChange={handleClickRadioButton1} checked={radio1.includes("F")===true ? true : false}/>False </label>
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

export default Linear;