import React, { useState } from "react";
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText} from "react-edit-text";
import "react-edit-text/dist/index.css";
import axios from 'axios';

const LeakyReLU = (props) => {

  var text_value = ''
  var radio1_value = ''

  var parmArr = String(props.params).split(' \n ')  // 파라미터별로 각각 분리

  for (var i=0; i<parmArr.length; i++){
        var param = String(parmArr[i]).replace('"', '');  // 쌍따옴표 제거  ex) 'p' : 0.5
        var eachParam = String(param).split(': ');  // 파라미터 이름과 값 분리  ex) ['p', 0.5]

        switch(i){  // 파라미터별로 해당 객체 값 설정
            case 0:  // 'kernel_size': (3, 3) 이므로, 괄호 안에서 3과 3을 따로 분리해주어야함
                text_value = String(eachParam[1]);
                break;
            case 1:
                radio1_value = String(eachParam[1]);
                break;
        }
    }

  const [text, setText] = React.useState(text_value);
  const [radio1, setRadio1] = React.useState(radio1_value);

  const { open, save, header } = props;

  const handleClickRadioButton1 = (e) => {
    console.log(e.target.value)
    setRadio1(e.target.value)
  }

  const bfsave=(event)=>{

    //console.log('radio1', radio1);
  
    console.log('props.params', props.params)
  
      var send_message = "'negative_slope': ".concat(text).concat(" \n 'inplace': ").concat(radio1)
  
      console.log(send_message);
      // node update하기 ********************
      axios.put("/api/node/".concat(String(props.layer).concat('/')),{
          order: String(props.layer),
          layer: "LeakyReLU",
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
                setText("0.01");
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
              <div>
                <label htmlFor="text">negative_slope:</label>
                <EditText
                  name="negative_slope"
                  type="number"
                  style={{ width: "40px" }}
                  value={text}
                  onChange={setText}
                  inline
                />
              </div>
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

export default LeakyReLU;