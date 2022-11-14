import React, { useState, useRef, useEffect } from "react";
import "../styles.css";
import Sidebar from "./sidebar";
import CustomEdge from "./CustomEdge";
import EditModal from "./layer/PopupModal";
import MaxPoolModal from "./layer/MaxPool";
import AvgPool2d from "./layer/AvgPool2d";
import AdaptiveAvgPool2d from "./layer/AdaptiveAvgPool2d";
import BatchNorm2d from "./layer/BatchNorm2d";
import Linear from "./layer/Linear";
import Dropout from "./layer/Dropout";
import ConstantPad2d from "./layer/ConstantPad2d";
import BCELoss from "./layer/BCELoss";
import LeakyReLU from "./layer/LeakyReLU";
import ReLU from "./layer/ReLU";
import ReLU6 from "./layer/ReLU6";
import Sigmoid from "./layer/Sigmoid";
import Softmax from "./layer/Softmax";
import Tanh from "./layer/Tanh";
import ZeroPad2d from "./layer/ZeroPad2d";
import CrossEntropyLoss from "./layer/CrossEntropyLoss";
import MSELoss from "./layer/MSELoss";
import Flatten from "./layer/Flatten";
import Upsample from "./layer/Upsample";
import axios from 'axios';
import yolo from '../img/YOLOv5.png';
import { initialArch } from '../initialArch';

import ReactFlow, {
  addEdge,
  MiniMap,
  ReactFlowProvider,
  removeElements
} from "react-flow-renderer";
import GenerateButton from "./GenerateButton";

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

function BasicGraph() {
  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [elements, setElements] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);



  if(checkFirst == 0){
      console.log('실행')
      axios.post("/api/running/",{     // status_report에 started 저장 (메인페이지 첫 실행시)
            timestamp: Date.now(),
            msg: 'started'
          }).then(function(response){
            console.log(response)
          }).catch(err=>console.log(err));
            // Initializate selected architecture
    var initElement = initialArch();
    for (var i=0;i<initElement.length;i++) {
      elements.push(initElement[i]);
      // setElements((es) => es.concat(initElement[i]));
    }
    checkFirst=1;
  }

  const notRunningState = setInterval(()=>{
////    console.log("[post] 동작 중지");
//    running_id += 1;
    axios.post("/api/status_report/", {

      timestamp: Date.now(),
//      running: 0,
    }).then(function(response){
        //console.log(timestamp)
        })
        .catch(e => console.log(e));
    }, initRunningStateTime * 1000)

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


  const onRunningStateClick = (e) => {
    e.preventDefault();
    clearInterval(notRunningState);
    //onRunningState();
    clearInterval(notRunningState);
    notRunningState();
  };

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

      axios.post("/api/edge/",{
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
//    console.log(nowc);
  };
  const onDrop = async (event) => {
    event.preventDefault();
    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
    const name = event.dataTransfer.getData("application/reactflow");
    const subp = event.dataTransfer.getData("subparameters");
    const position = reactFlowInstance.project({
      x: event.clientX - reactFlowBounds.left,
      y: event.clientY - reactFlowBounds.top -360
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
        background: "#434854",
        width: 150,
        color: "#fff",
        fontSize: "20px",
        fontFamily: "Helvetica",
        boxShadow: "5px 5px 5px 0px rgba(0,0,0,.10)"
      },
      data: {
        label: `${name}`,
        subparam: `${subp}`
      }
    };
    setElements((es) => es.concat(newNode));
  };

  const C = () => {
    if (state === "Conv2d")
      return (
        <EditModal
          params={nowp}
          layer={nowc}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
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
          header="Node"
        ></Upsample>
      );
  };

  return (
    <div className="dndflow" onClick={onRunningStateClick}>
      <ReactFlowProvider>

        <div
          className="reactflow-wrapper"
          style={{ height: "60vh", width: "950px" }}
          ref={reactFlowWrapper}
        >
        <tc>
          <img src={yolo} alt="model image"/>
        </tc>

        <br/>
          <ReactFlow 
//            onClick={onRunningStateClick}
            initElement={initialArch}
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
            onNodeMouseEnter={onNodeClick}
            onElementsRemove={onElementsRemove}
          >
            <C />
          </ReactFlow>

          <button class="arrange">정렬</button>
          <GenerateButton elements={elements} > </GenerateButton>

          <MiniMap
            nodeStrokeColor={(n) => {
              if (n.style?.background) return n.style.background;
              if (n.type === "input") return "#fff";
              if (n.type === "output") return "#ff0072";
              if (n.type === "default") return "#1a192b";
              return "#eee";
            }}
            nodeColor={(n) => {
              if (n.style?.background) return n.style.background;

              return "#fff";
            }}
            nodeBorderRadius={2}
          />
        </div>
        <Sidebar />
      </ReactFlowProvider>

    </div>
  );
}

export default function Flow() {
  return <BasicGraph />;
}
