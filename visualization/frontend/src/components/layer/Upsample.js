import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar";
import axios from 'axios';



const Upsample = (props) => {
  console.log('String(props.params)', String(props.params));

  var text_value = ''  // 변수 선언
  var text2_value = ''
  var text3_value = ''
  var text4_value = ''
  var text5_value = ''

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
            text3_value = eachParam[1];
            break;
        case 3:
            text4_value = eachParam[1];
            break;
        case 4:
            text5_value = eachParam[1];
            break;

    }
  }

  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);
  const [text3, setText3] = React.useState(text3_value);
  const [text4, setText4] = React.useState(text4_value);
  const [text5, setText5] = React.useState(text5_value);



  const { open, save, close, header } = props;

  const bfsave=(event)=>{


    var send_message = "'size': ".concat(text)
        .concat(" \n 'scale factor': ").concat(text2)
        .concat(" \n 'mode': ").concat(text3)
        .concat(" \n 'align corners': ").concat(text4)
        .concat(" \n 'recompute scale factor': ").concat(text5)



    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "Upsample",
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
              setText('None')
              setText2('None')
              setText3('Nearest')
              setText4('None')
              setText5('None')}
  } >
              default
            </button>
            <button className="save" onClick={bfsave}>
              save
            </button>

          </header>
          <main>
          <React.Fragment>


          <div><label htmlFor="text">size:</label>
          <EditText name="size" type="text" style={{width: '50px'}} value={text}
            onChange={setText} inline/></div>
          <div><label htmlFor="text">scale factor:</label>
          <EditText name="scale_factor" type="text" style={{width: '50px'}} value={text2}
            onChange={setText2} inline/></div>





          <div><label htmlFor="text">mode:</label>
          <EditText name="mode" type="text" style={{width: '60px'}} value={text3}
            onChange={setText3} inline/></div>

          <div><label htmlFor="text">align corners:</label>
          <EditText name="align_corners" type="text" style={{width: '50px'}} value={text4}
            onChange={setText4} inline/></div>

           <div><label htmlFor="text">recompute scale factor:</label>
          <EditText name="recompute_scale_factor" type="text" style={{width: '50px'}} value={text5}
            onChange={setText5} inline/></div>
          </React.Fragment>
          </main>

        </section>
      ) : null}
    </div>
  );
};

export default Upsample;