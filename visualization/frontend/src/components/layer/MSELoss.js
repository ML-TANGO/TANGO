import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar";
import axios from 'axios';



const MSELoss = (props) => {

  console.log('String(props.params)', String(props.params));

  var radio1_value = ''
  var radio2_value = ''
  var text_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

  for (var i=0; i<parmArr.length; i++){
    var param = parmArr[i].replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
    var eachParam = param.split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

    switch(i){  // 파라미터별로 해당 객체 값 설정
        case 0:
            radio1_value = eachParam[1];
            break;
        case 1:
            radio2_value = eachParam[1];
            break;
        case 2:
            text_value = eachParam[1];
            break;


    }
  }


  const [radio1, setRadio1] = React.useState(radio1_value);
  const [radio2, setRadio2] = React.useState(radio2_value);
  const [text, setText] = React.useState(text_value);



  const { open, save, close, header } = props;


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

  //console.log('props.params', props.params)

    var send_message = "'size_average': ".concat(radio1)
        .concat(" \n 'reduce': ").concat(radio2)
        .concat(" \n 'reduction': ").concat(text)


    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "MSELoss",
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
              setRadio1('True')
              setRadio2('True')
              setText('mean')}
  } >
              default
            </button>
            <button className="save" onClick={bfsave}>
              save
            </button>

          </header>
          <main>
          <React.Fragment>



          <div><label htmlFor="text">size_average:</label>
          <label> <input type="radio" name="radio1" value="True" onChange={handleClickRadioButton1} checked={radio1.includes("T")===true ? true : false}/>True </label>
          <label> <input type="radio" name="radio1" value="False" onChange={handleClickRadioButton1} checked={radio1.includes("F")===true ? true : false}/>False </label>
          </div>

          <div><label htmlFor="text">reduce:</label>
          <label> <input type="radio" name="radio2" value="True" onChange={handleClickRadioButton2} checked={radio2.includes("T")===true ? true : false}/>True </label>
          <label> <input type="radio" name="radio2" value="False" onChange={handleClickRadioButton2} checked={radio2.includes("F")===true ? true : false}/>False </label>
          </div>

          <div><label htmlFor="text">reduction:</label>
          <EditText name="reduction" type="text" style={{width: '50px'}} value={text}
            onChange={setText} inline/></div>



          </React.Fragment>
          </main>

        </section>
      ) : null}
    </div>
  );
};

export default MSELoss;