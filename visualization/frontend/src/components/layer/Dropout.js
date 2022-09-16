import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar";
import axios from 'axios';



const Dropout = (props) => {

  var text_value = ''  // 변수 선언
  var radio1_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리
  //console.log(parmArr);

  for (var i=0; i<parmArr.length; i++){
    var param = parmArr[i].replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
    var eachParam = param.split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

    switch(i){  // 파라미터별로 해당 객체 값 설정
        case 0:
            text_value = eachParam[1];
            break;
        case 1:
            radio1_value = eachParam[1];
            break;
    }
  }

  const [text, setText] = React.useState(text_value);
  const [radio1, setRadio1] = React.useState(radio1_value);

  //라디오 버튼 값 조회
  const handleClickRadioButton1 = (e) => {
    console.log(e.target.value)
    setRadio1(e.target.value)
  }

  const { open, save, close, header } = props;
  const bfsave=(event)=>{

  console.log('props.params', props.params)

    var send_message = "'p': ".concat(text).concat(" \n 'inplace': ").concat(radio1)
        //.concat(") \n 'dilation': ").concat(text7).concat(" \n 'return_indices': False \n 'ceil_mode': False")

    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "Dropout",
        parameters: send_message
    }).then(function(response){
        console.log(response)
    }).catch(err=>console.log(err));

    save();
  };

  // const bfdelete=(event)=>{

  //   axios.delete("/api/node/".concat(String(props.layer).concat('/')))
  // .then(function (response) {
  //       console.log(response)
  // })
  // .catch(function (error) {
  //   // handle error
  // })
  // .then(function () {
  //   // always executed
  // });
  // close();
  // };

  return (
    <div className={open ? 'openModal modal' : 'modal'}>
      {open ? (
        <section>
          <header>
            {header}


            <button className="close" onClick={() => {
              setText('0.5')}
  } >
              default
            </button>
            <button className="save" onClick={bfsave}>
              save
            </button>

          </header>
          <main>
          <React.Fragment>


          <div><label htmlFor="text">p:</label>
          <EditText name="p" type="text" style={{width: '50px'}} value={text}
            onChange={setText} inline/></div>


          <div><label htmlFor="text">inplace:</label>
          <label> <input type="radio" name="radio1" value="True" onChange={handleClickRadioButton1} checked={radio1.includes("T")===true ? true : false}/>True </label>
          <label> <input type="radio" name="radio1" value="False" onChange={handleClickRadioButton1} checked={radio1.includes("F")===true ? true : false}/>False </label>
          </div>


          </React.Fragment>
          </main>

        </section>
      ) : null}
    </div>
  );
};

export default Dropout;