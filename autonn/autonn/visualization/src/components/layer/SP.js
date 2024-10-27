import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar/LayerToggle";
import axios from 'axios';



const SP = (props) => {
  console.log('String(props.params)', String(props.params));

  var text_value = ''  // 변수 선언
  // var text2_value = ''
  var text3_value = ''
  // var text4_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

  for (var i=0; i<parmArr.length; i++){
    var param = parmArr[i].replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
    var eachParam = param.split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

    switch(i){  // 파라미터별로 해당 객체 값 설정
      case 0:  // 'kernel_size': (3, 3) 이므로, 괄호 안에서 3과 3을 따로 분리해주어야함
          var kernelArray = String(eachParam[1]).split(', ');
          text_value = String(kernelArray[0]).replace("(", "");
          // text2_value = String(kernelArray[1]).replace(")", "");
          break;
      case 1:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
          var strideArray = String(eachParam[1]).split(', ');
          text3_value = String(strideArray[0]).replace("(", "");
          // text4_value = String(strideArray[1]).replace(")", "");
          break;
    }
  }

  const [text, setText] = React.useState(text_value);
  // const [text2, setText2] = React.useState(text2_value);
  const [text3, setText3] = React.useState(text3_value);
  // const [text4, setText4] = React.useState(text4_value);

  const { open, save, close, header } = props;

  const bfsave=(event)=>{
    //console.log('props.params', props.params)
    var send_message = "'kernel_size': (".concat(text).concat(', ').concat(text)
        .concat(") \n 'stride': (").concat(text3).concat(', ').concat(text3).concat(")")

    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
      order: String(props.layer),
      layer: "SP",
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
                <label htmlFor="text">kernel_size:</label>
                <EditText name="kernel_size1" type="number" style={{width: '40px'}} value={text}
          onChange={setText} inline/>
                <label htmlFor="text"> X </label>
                <EditText name="kernel_size2" type="number" style={{width: '40px'}} value={text}
          onChange={setText} inline/>
            </li>
            <li>
                <label htmlFor="text">stride:</label>
                <EditText name="stride1" type="number" style={{width: '40px'}} value={text3}
          onChange={setText3} inline/>
                <label htmlFor="text"> X </label>
                <EditText name="stride2" type="number" style={{width: '40px'}} value={text3}
          onChange={setText3} inline/>
            </li>
          </div>

          </React.Fragment>
          </main>
          <div className="btnDiv">
            <button className="close" onClick={() => {
              setText('3')
              // setText2('3')
              setText3('1')
              // setText4('1')
            }}>
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

export default SP;
