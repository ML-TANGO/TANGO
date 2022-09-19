import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar";
import axios from 'axios';



const Flatten = (props) => {
  console.log('String(props.params)', String(props.params));

  var text_value = ''  // 변수 선언
  var text2_value = ''

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

    }
  }

  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);





  const { open, save, close, header } = props;


  const bfsave=(event)=>{

  //console.log('radio1', radio1);

  //console.log('props.params', props.params)

    var send_message = "'start dim': ".concat(text)
        .concat(" \n 'end dim': ").concat(text2)


    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "Flatten",
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
    <div className={open ? 'openModal modal' : 'modal'}>
      {open ? (
        <section>
          <header>
            {header}


            <button className="close" onClick={() => {
              setText('1')
              setText2('-1')}
  }>
              default
            </button>
            <button className="save" onClick={bfsave}>
              save
            </button>

          </header>
          <main>
          <React.Fragment>


          <div><label htmlFor="text">start dim:</label>
          <EditText name="start_dim" type="number" style={{width: '50px'}} value={text}
            onChange={setText} inline/></div>
          <div><label htmlFor="text">end dim:</label>
          <EditText name="end_dim" type="number" style={{width: '50px'}} value={text2}
            onChange={setText2} inline/></div>

          </React.Fragment>
          </main>

        </section>
      ) : null}
    </div>
  );
};

export default Flatten;
