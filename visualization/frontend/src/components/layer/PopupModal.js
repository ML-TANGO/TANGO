import React, { useState } from 'react';
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar";


import axios from 'axios';

const EditModal = (props) => {

  var text_value = ''  // 변수 선언
  var text2_value = ''
  var text3_value = ''
  var text4_value = ''
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
                var kernelArray = String(eachParam[1]).split(', ');
                text3_value = String(kernelArray[0]).replace("(", "");
                text4_value = String(kernelArray[1]).replace(")", "");
                break;
            case 3:  // 'stride': (1, 1) 이므로, 괄호 안에서 1과 1을 따로 분리해주어야함
                var strideArray = String(eachParam[1]).split(', ');
                text5_value = String(strideArray[0]).replace("(", "");
                text6_value = String(strideArray[1]).replace(")", "");
                break;
            case 4:  // 'padding': (0, 0) 이므로, 괄호 안에서 0과 0을 따로 분리해주어야함
                var paddingArray = String(eachParam[1]).split(', ');
                text7_value = String(paddingArray[0]).replace("(", "");
                text8_value = String(paddingArray[1]).replace(")", "");
        }
    }

  const [text, setText] = React.useState(text_value);
  const [text2, setText2] = React.useState(text2_value);
  const [text3, setText3] = React.useState(text3_value);
  const [text4, setText4] = React.useState(text4_value);
  const [text5, setText5] = React.useState(text5_value);
  const [text6, setText6] = React.useState(text6_value);
  const [text7, setText7] = React.useState(text7_value);
  const [text8, setText8] = React.useState(text8_value);


  // 수정된 부분
  const { open, save, close, header } = props;

  const bfsave=(event)=>{

    var send_message = "'in_channels': ".concat(text)
        .concat(" \n 'out_channels': ").concat(text2)
        .concat(" \n 'kernel_size': (").concat(text3).concat(', ').concat(text4)
        .concat(") \n 'stride': (").concat(text5).concat(', ').concat(text6)
        .concat(") \n 'padding': (").concat(text7).concat(', ').concat(text8).concat(')')

    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "Conv2d",
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

 const bfdelete=(event)=>{

    axios.delete("/api/node/".concat(String(props.layer).concat('/')))
  .then(function (response) {
        console.log(response)
  })
  .catch(function (error) {
    // handle error
  })
  .then(function () {
    // always executed
  });
  close();
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
              setText('3')
              setText2('64')
              setText3('3')
              setText4('3')
              setText5('1')
              setText6('1')
              setText7('1')
              setText8('1')
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


          <div><label htmlFor="text">in_channels:</label>
          <EditText name="in_channels" type="number" style={{width: '50px'}} value={text}
            onChange={setText} inline/></div>
          <div><label htmlFor="text">out_channels:</label>
          <EditText name="out_channels" type="number" style={{width: '50px'}} value={text2}
            onChange={setText2} inline/></div>
            <div><label htmlFor="text">kernel_size:</label>
          <EditText name="kernel_size1" type="number" style={{width: '40px'}} value={text3}
            onChange={setText3} inline/>
          <label htmlFor="text"> X </label>
          <EditText name="kernel_size2" type="number" style={{width: '40px'}} value={text4}
            onChange={setText4} inline/>
          </div>
           <div><label htmlFor="text">stride:</label>
          <EditText name="stride1" type="number" style={{width: '40px'}} value={text5}
            onChange={setText5} inline/>
          <label htmlFor="text"> X </label>
          <EditText name="stride2" type="number" style={{width: '40px'}} value={text6}
            onChange={setText6} inline/>
          </div>
          <div><label htmlFor="text">padding:</label>
          <EditText name="padding1" type="number" style={{width: '40px'}} value={text7}
            onChange={setText7} inline/>
          <label htmlFor="text"> X </label>
          <EditText name="padding2" type="number" style={{width: '40px'}} value={text8}
            onChange={setText8} inline/>
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

export default EditModal;