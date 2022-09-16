import React, { useState } from "react";
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea } from "react-edit-text";
import "react-edit-text/dist/index.css";
import axios from 'axios';

const ReLU6 = (props) => {
  var radio1_value = ''  // 변수 선언

  var parmArr = String(props.params).split(', \n ')  // 파라미터별로 각각 분리

    for (var i=0; i<parmArr.length; i++){
        var param = parmArr[i].replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
        var eachParam = param.split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

        switch(i){  // 파라미터별로 해당 객체 값 설정
            case 0:
                radio1_value = eachParam[1];
                break;
        }
    }

  const [radio1, setRadio1] = React.useState(radio1_value);
  const { open, save, header } = props;
  const handleClickRadioButton1 = (e) => {
    //console.log(e.target.value)
    setRadio1(e.target.value)
  }
  console.log('props.params', props.params)


  const bfsave=(event)=>{

  //console.log('radio1', radio1);

  console.log('props.params', props.params)

    var send_message = "'inplace': ".concat(radio1)
    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "ReLU6",
        parameters: send_message
    }).then(function(response){
        console.log(response)
    }).catch(err=>console.log(err));
    // node update하기 ********************


    save();
  };



  return (
    <div className={open ? "openModal modal" : "modal"}>
      {open ? (
        <section>
          <header>
            {header}
            {/* { <button className="save" onClick={close}>
              Save
            </button> } */}

            <button
              className="close"
              onClick={() => {
                setRadio1('False')
              }}
            >
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
            <div><label htmlFor="text">inplace:</label>
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

export default ReLU6;