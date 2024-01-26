import React, { useState, useRef, useEffect } from "react";
import "../../styles.css";
import CustomEdge from "../CustomEdge";
import EditModal from "../layer/PopupModal";
import MaxPoolModal from "../layer/MaxPool";
import AvgPool2d from "../layer/AvgPool2d";
import AdaptiveAvgPool2d from "../layer/AdaptiveAvgPool2d";
import BatchNorm2d from "../layer/BatchNorm2d";
import Linear from "../layer/Linear";
import Dropout from "../layer/Dropout";
import ConstantPad2d from "../layer/ConstantPad2d";
import BCELoss from "../layer/BCELoss";
import LeakyReLU from "../layer/LeakyReLU";
import ReLU from "../layer/ReLU";
import ReLU6 from "../layer/ReLU6";
import Sigmoid from "../layer/Sigmoid";
import Softmax from "../layer/Softmax";
import Tanh from "../layer/Tanh";
import ZeroPad2d from "../layer/ZeroPad2d";
import CrossEntropyLoss from "../layer/CrossEntropyLoss";
import MSELoss from "../layer/MSELoss";
import Flatten from "../layer/Flatten";
import Upsample from "../layer/Upsample";
import InitialArch from '../../InitialArch';
import axios from 'axios';
import ReactFlow, {
  addEdge,
  MiniMap,
  ReactFlowProvider,
  removeElements,
  Controls, ControlButton,

} from "react-flow-renderer";
import GenerateButton from "../GenerateButton";
import Tab from "../sidebar/Tab";
import NetworkInformation from "../sidebar/NetworkInformation";
import arange_icon from "../../img/swap.png";

let id = 1;
const getId = () => `${id}`;
let nowc= 0;
const edgeTypes = {
  custom: CustomEdge
};
let nowp = "";
var checkFirst = 0;
let initRunningStateTime = 100;
var running_id = 0;

function InfoList() {
  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [elements, setElements] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);


  //
  if(checkFirst == 0){
      console.log('실행')
      // axios.post("/api/running/",{     // status_report에 started 저장 (메인페이지 첫 실행시)
      //       timestamp: Date.now(),
      //       msg: 'started'
      //     }).then(function(response){
      //       console.log(response)
      //     }).catch(err=>console.log(err));
      //       // Initializate selected architecture
    var initElement = InitialArch();
    for (var i=0;i<initElement.length;i++) {
      elements.push(initElement[i]);
      // setElements((es) => es.concat(initElement[i]));
    }
    checkFirst=1;
  }

//   const notRunningState = setInterval(()=>{
// ////    console.log("[post] 동작 중지");
// //    running_id += 1;
//     axios.post("/api/status_report/", {
//
//       timestamp: Date.now(),
// //      running: 0,
//     }).then(function(response){
//         //console.log(timestamp)
//         })
//         .catch(e => console.log(e));
//     }, initRunningStateTime * 1000)

//  const onRunningState = (()=>{
////    console.log("[post] 동작 중");
//
//    running_id += 1;
//    axios.post("/api/running/", {
//      id : running_id,
//      running: 1,
//    }).then(function(response){
//      console.log(response)
//      })
//      .catch(e => console.log(e));
//  })

  //
  // const onRunningStateClick = (e) => {
  //   e.preventDefault();
  //   clearInterval(notRunningState);
  //   //onRunningState();
  //   clearInterval(notRunningState);
  //   notRunningState();
  // };

  const onConnect = async (params) => {
    setElements((els) => addEdge(params, els));
      // edge create **********************

      const get_edge = async () => {
        try {
          return await axios.get('/api/edge/');
        } catch (error) {
          console.error(error);
        }
      };
      const cedge = await get_edge();
      var maxId = 0;
      for(var i=0; i<cedge.data.length; i++){
       if(maxId<cedge.data[i].id){
        maxId = cedge.data[i].id
       }
      }
      axios.post("/api/edge/",{
        id: maxId+1,
        id: cedge.data.length+1,
        prior: params.source,
        next: params.target
      }).then(function(response){
        console.log(response)
      }).catch(err=>console.log(err));
  };

  const onDeleteEdge = (e) => {
    console.log(e.target);
  }

  const onLoad = (rFInstance) => setReactFlowInstance(rFInstance);

  const onElementsRemove = (remove) => {
    setElements((els)=>removeElements(remove, els));
    deleteModal(remove);
  }

  //default 값을 못받아오는 이유??
  const openModal = async () => {
    const get_params = async () => {
      try {
        await axios.get('/api/node/'.concat(String(nowc)).concat('/')).then((response) => {
          nowp = response.data.parameters;
        });
      } catch (error) {
        console.error(error);
      }
    };
    await get_params();
    await setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
  };

  const saveModal = () => {
    setModalOpen(false);
  };

  const deleteModal = (remove) => {
    axios.get("/api/node/".concat(String(nowc)).concat('/'))
    .then(function(response){
    console.log(response)});
    console.log("remove", remove)
    if(remove[0].data){
        console.log('node')
        axios.delete("/api/node/".concat(String(nowc)).concat('/'));
        axios.get("/api/edge/")
        .then(function(response){
        for(var i=0;i<response.data.length;i++){
            if(String(response.data[i].prior) === String(nowc)){
                axios.delete("/api/edge/".concat(String(response.data[i].id)).concat('/'));
            }
            if(String(response.data[i].next) === String(nowc)){
                axios.delete("/api/edge/".concat(String(response.data[i].id)).concat('/'));
            }
        }
        });
    } else{
    console.log('edge')
    axios.get("/api/edge/")
    .then(function(response){
      for(var i=0;i<response.data.length;i++){
        if(String(response.data[i].prior) === String(remove[0].source)){
          if(String(response.data[i].next) === String(remove[0].target)){
            axios.delete("/api/edge/".concat(String(response.data[i].id)).concat('/'));
          }
        }
      }
  });
    }
  };

  const onDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  };
  const [state, setState] = useState("");
  const onNodeClick = (event, node) => {
    setState(node.data.label);
    nowc = node.id;
   console.log(nowc);
   console.log(state);
  };

  const onDrop = async (event) => {
    event.preventDefault();
    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
    const name = event.dataTransfer.getData("application/reactflow");
    const color = event.dataTransfer.getData('colorNode');
    const subp = event.dataTransfer.getData("subparameters");
    const position = reactFlowInstance.project({
      x: event.clientX - reactFlowBounds.left - 72,
      y: event.clientY - reactFlowBounds.top - 10
    });
    const get_node = async () => {
      try {
        return await axios.get('/api/node/');
      } catch (error) {
        console.error(error);
      }
    };

    const cnode = await get_node();

    // cnode의 order값이 가장 큰 값 탐색
    var maxOrder = 0;
    for(var i=0; i<cnode.data.length; i++){
        if(maxOrder<cnode.data[i].order){
            maxOrder = cnode.data[i].order
        }
    }

    // 가장 큰 order+1로 id값 설정
    const nid = maxOrder+1;
    id = nid

    //node create **********************
    //const cnode = plusId()
    axios.post("/api/node/",{
        order: id,
        layer: name,
        parameters: subp
    }).then(function(response){
        console.log(response)
    }).catch(err=>console.log(err));
    //node create **********************


    const newNode = {
      id: getId(),
      type: "default",
      position,
      style: {
        background: `${color}`,
        width: 135,
        fontSize: "20px",
        fontFamily: "Helvetica",
        // boxShadow: "5px 5px 5px 0px rgba(0,0,0,.10)",
        boxShadow: "7px 7px 7px 0px rgba(0, 0, 0, 0.2)",
        borderRadius: "10px",
        border: "none"
      },
      data: {
        label: `${name}`,
        subparam: `${subp}`
      }
    };
    setElements((es) => es.concat(newNode));
  };

  const   C = () => {
    if (state === "Conv2d")
      return (
        <EditModal
          params={nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></EditModal>
      );
    if (state === "MaxPool2d")
      return (
        <MaxPoolModal
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></MaxPoolModal>
      );
    if (state === "AvgPool2d")
      return (
        <AvgPool2d
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></AvgPool2d>
      );
    if (state === "AdaptiveAvgPool2d (ResNet)")
      return (
        <AdaptiveAvgPool2d
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></AdaptiveAvgPool2d>
      );
     if (state === "Softmax")
      return (
        <Softmax
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></Softmax>
      );
    if (state === "ConstantPad2d")
      return (
        <ConstantPad2d
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></ConstantPad2d>
        );
    if (state === "BatchNorm2d")
      return (
        <BatchNorm2d
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></BatchNorm2d>
      );

    if (state === "MSELoss")
      return (
        <MSELoss
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></MSELoss>
      );
    if (state === "Tanh")
      return (
        <Tanh
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></Tanh>
      );
    if (state === "Sigmoid")
      return (
        <Sigmoid
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></Sigmoid>
      );
    if (state === "CrossEntropyLoss")
      return (
        <CrossEntropyLoss
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></CrossEntropyLoss>
      );
    if (state === "Linear")
      return (
        <Linear
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></Linear>
      );
    if (state === "Dropout")
      return (
        <Dropout
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></Dropout>
      );
      if (state === "ZeroPad2d")
      return (
        <ZeroPad2d
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></ZeroPad2d>
      );
      if (state === "BCELoss")
      return (
        <BCELoss
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></BCELoss>
      );
      if (state === "LeakyReLU")
      return (
        <LeakyReLU
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></LeakyReLU>
      );
       if (state === "ReLU")
      return (
        <ReLU
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></ReLU>
      );
      if (state === "ReLU6")
      return (
        <ReLU6
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></ReLU6>
      );
       if (state === "Flatten")
      return (
        <Flatten
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></Flatten>
      );

    else
      return (
        <Upsample
          params = {nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
        ></Upsample>
      );
  };

  return (
      <div className="FullPage">
        <div className="Sidebar">
          <Tab/>
          <NetworkInformation/>
          <div className="LayerInfo">
            <h3>Layer Information</h3>
            {/*<div className="Modal">*/}
              <C />
            {/*</div>*/}
         </div>
        </div>

    <div className="dndflow" >
      <ReactFlowProvider>
        <div className="reactflow-wrapper" ref={reactFlowWrapper}>
          <ReactFlow
//            onClick={onRunningStateClick}
            initElement={InitialArch}
            onConnect={onConnect}
            elements={elements}
            onLoad={onLoad}
            onDrop={onDrop}
            onDragOver={onDragOver}
            snapToGrid={true}
            edgeTypes={edgeTypes}
            key="edges"
            onNodeDoubleClick={openModal}
            onEdgeDoubleClick={onDeleteEdge}
            onElementsRemove={onElementsRemove}
            onElementClick={onNodeClick}

          >
            <Controls showZoom="" showInteractive="" showFitView="">
              <ControlButton onClick={() => console.log('action')} title="action">
                <img src={arange_icon}/>
              </ControlButton>
            </Controls>

            <button className="inspect">Inspect</button>
            <GenerateButton elements={elements} />

          </ReactFlow>
        </div>
        </ReactFlowProvider>
      </div>
  </div>
  );
}

export default function Info() {
  return <InfoList />;
}
