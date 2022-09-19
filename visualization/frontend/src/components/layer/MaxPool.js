import React, { useState } from 'react';
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar";

// *** 이거 추가해야 함
import axios from 'axios';
const MaxPoolModal = (props) => {

  console.log('props', props);

  var text_value = ''  // 변수 선언
  var text2_value = ''
  var text3_value = ''
  var text4_value = ''
  var text5_value = ''
  var text6_value = ''
  var text7_value = ''
  var radio1_value = ''
  var radio2_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

  for (var i=0; i<parmArr.length; i++){
        var param = String(parmArr[i]).replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
        var eachParam = String(param).split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

        switch(i){  // 파라미터별로 해당 객체 값 설정
            case 0:  // 'kernel_size': (3, 3) 이므로, 괄호 안에서 3과 3을 따로 분리해주어야함
                var kernelArray = String(eachParam[1]).split(', ');
                text_value = String(kernelArray[0]).replace("(", "");
                text2_value = String(kernelArray[1]).replace(")", "");
                break;
            case 1:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                var strideArray = String(eachParam[1]).split(', ');
                text3_value = String(strideArray[0]).replace("(", "");
                text4_value = String(strideArray[1]).replace(")", "");
                break;
            case 2:  // 'padding': (0, 0) 이므로, 괄호 안에서 0과 0을 따로 분리해주어야함
                var paddingArray = String(eachParam[1]).split(', ');
                text5_value = String(paddingArray[0]).replace("(", "");
                text6_value = String(paddingArray[1]).replace(")", "");
            case 3:
                text7_value = String(eachParam[1]);
                break;
            case 4:
                radio1_value = String(eachParam[1]);
                break;
            case 5:
                radio2_value = String(eachParam[1]);
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

  const [radio1, setRadio1] = React.useState(radio1_value);
  const [radio2, setRadio2] = React.useState(radio2_value);

  // 수정된 부분
  const { open, save, close, header } = props;

//  let isCheckedrt
//  let isCheckedct
//  let isCheckedrf
//  let isCheckedcf
//
  //라디오 버튼 값 조회
  const handleClickRadioButton1 = (e) => {
    console.log(e.target.value)
    setRadio1(e.target.value)
  }

    //라디오 버튼 값 조회
  const handleClickRadioButton2 = (e) => {
    console.log(e.target.value)
    setRadio2(e.target.value)
  }


  const bfsave=(event)=>{

  //console.log('radio1', radio1);

  console.log('props.params', props.params)

    var send_message = "'kernel_size': (".concat(text).concat(', ').concat(text2)
        .concat(") \n 'stride': (").concat(text3).concat(', ').concat(text4)
        .concat(") \n 'padding': (").concat(text5).concat(', ').concat(text6)
        .concat(") \n 'dilation': ").concat(text7).concat(" \n 'return_indices': ").concat(radio1).concat(" \n  'ceil_mode': ").concat(radio2)
        //.concat(") \n 'dilation': ").concat(text7).concat(" \n 'return_indices': False \n 'ceil_mode': False")

    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "MaxPool2d",
        parameters: send_message
    }).then(function(response){
        console.log(response)
    }).catch(err=>console.log(err));
    // node update하기 ********************

//    console.log(text8, "text7");
//    console.log(text9, "text7");

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

            <button className="close" onClick={() => {
              setText('2')
              setText2('2')
              setText3('2')
              setText4('2')
              setText5('0')
              setText6('0')
              setText7('1')
              setRadio1('False')
              setRadio2('False')
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



          <div><label htmlFor="text">kernel_size:</label>
          <EditText name="kernel_size1" type="number" style={{width: '40px'}} value={text}
            onChange={setText} inline/>
          <label htmlFor="text"> X </label>
          <EditText name="kernel_size2" type="number" style={{width: '40px'}} value={text2}
            onChange={setText2} inline/>
          </div>
          <div><label htmlFor="text">stride:</label>
          <EditText name="stride1" type="number" style={{width: '40px'}} value={text3}
            onChange={setText3} inline/>
          <label htmlFor="text"> X </label>
          <EditText name="stride2" type="number" style={{width: '40px'}} value={text4}
            onChange={setText4} inline/>
          </div>

          <div><label htmlFor="text">padding:</label>
          <EditText name="padding1" type="number" style={{width: '40px'}} value={text5}
            onChange={setText5} inline/>
          <label htmlFor="text"> X </label>
          <EditText name="padding2" type="number" style={{width: '40px'}} value={text6}
            onChange={setText6} inline/>
          </div>
          <div><label htmlFor="text">dilation:</label>
          <EditText name="in_channels" type="number" style={{width: '50px'}} value={text7}
            onChange={setText7} inline/></div>

          <div><label htmlFor="text">return_indices:</label>
          <label> <input type="radio" name="radio1" value="True" onChange={handleClickRadioButton1} checked={radio1.includes("T")===true ? true : false}/>True </label>
          <label> <input type="radio" name="radio1" value="False" onChange={handleClickRadioButton1} checked={radio1.includes("F")===true ? true : false}/>False </label>
          </div>

          <div><label htmlFor="text">ceil_mode:</label>
          <label> <input type="radio" name="radio2" value="True" onChange={handleClickRadioButton2} checked={radio2.includes("T")===true ? true : false}/>True </label>
          <label> <input type="radio" name="radio2" value="False" onChange={handleClickRadioButton2} checked={radio2.includes("F")===true ? true : false}/>False </label>
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

export default MaxPoolModal;