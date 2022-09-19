import React, { useState } from "react";
//import ReactModal from 'react-modal';
import "./ModalStyle.css";
import { EditText, EditTextarea } from "react-edit-text";
import "react-edit-text/dist/index.css";

const Sigmoid = (props) => {


  const { open, save, header } = props;

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

              }}
            >
              default
            </button>
            <button className="save" onClick={save}>
              save
            </button>
            {/* <button className="close" onClick={close}>
              &times;
            </button> */}
          </header>
          <main>
            <React.Fragment>
              <div></div>
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

export default Sigmoid;
