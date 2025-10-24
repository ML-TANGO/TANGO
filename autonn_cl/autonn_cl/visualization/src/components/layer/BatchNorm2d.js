import React, { useState } from 'react';
import "./ModalStyle.css";
import { EditText} from 'react-edit-text';
import 'react-edit-text/dist/index.css';
import Sidebar from "../sidebar/LayerToggle";
import axios from 'axios';



const BatchNorm2d = (props) => {
  const [text, setText] = React.useState(
    String(props.params).substr(16, 10)
  );

  const { open, save, close, header } = props;

  const bfsave=(event)=>{

  //console.log('radio1', radio1);

    var send_message = "'num_features': ".concat(text)
    console.log(send_message);
    // node update하기 ********************
    axios.put("/api/node/".concat(String(props.layer).concat('/')),{
        order: String(props.layer),
        layer: "BatchNorm2d",
        parameters: send_message
    }).then(function(response){
        console.log(response)
    }).catch(err=>console.log(err));
    // node update하기 ********************

//    console.log(text8, "text7");
//    console.log(text9, "text7");
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
                  <label htmlFor="text">num_features:</label>
                  <EditText name="num_features" type="number" style={{width: '50px'}} value={text}
                    onChange={setText} inline/>
              </li>
              </div>
          </React.Fragment>
          </main>
            <div className="btnDiv">
                <button className="close" onClick={() => {
              setText('512')}
  } >
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

export default BatchNorm2d;