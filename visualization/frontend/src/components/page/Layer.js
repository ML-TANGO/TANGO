import React, {useState, useRef, useEffect} from "react";

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
import BasicBlock from "../layer/BasicBlock";
import Bottleneck from "../layer/Bottleneck";
import axios from 'axios';
import { initialArch } from '../../initialArch';
import ReactFlow, {

  addEdge,
  MiniMap,
  ReactFlowProvider,
  removeElements,
    Controls,ControlButton
} from "react-flow-renderer";
import GenerateButton from "../GenerateButton";
import Tab from "../sidebar/Tab";
import LayerToggle from "../sidebar/LayerToggle";
import NetworkInformation  from "../sidebar/NetworkInformation";
import arange_icon from "../../img/swap.png";
import BasicBlockimg from "../../img/basicblock.png";
import BottleNeckimg from "../../img/bottleneck.png";

let id = 1;
const getId = () => `${id}`;
let nowc= 0;
const edgeTypes = {
  custom: CustomEdge
};
let nowp = "";
var checkFirst = 0;
let initRunningStateTime = 100;
//var running_id = 0;
var sortCount = 1;
var sortHeight = 0;
let sortList = [];
function LayerList() {
  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [elements, setElements] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [state, setState] = useState("");
  const [idState, setIdState] = useState("");
  const [paramState, setParam] = useState();
  var running_id = 0;

  useEffect(()=>{
    const get_params = async () => {
      try {
        await axios.get('/api/node/'.concat(String(idState)).concat('/')).then((response) => {
           setParam(response.data.parameters);
        });
      } catch (error) {
        console.error(error);
      }
    };
    get_params();
  },[idState]);





const onSortNodes = (sortList) => {




    console.log('back code');

    sortList = sortList.split(",");
    console.log(sortList);

  const sortedElements = elements.slice(); // elements 배열을 복사하여 새로운 배열을 생성합니다.
  console.log(sortedElements);
  console.log(' my code ');
  let sort_x_pos = 100 + sortCount;
  let sort_y_pos = 100 + sortCount;

   let isBlock = undefined;
  if(sortedElements[sortList[0]].sort !== "0"){
      isBlock = true;
  }else{
      isBlock = false;
    }


  for(var i = 0; i < sortList.length; i++) {
    for (var j = 0; j < sortedElements.length; j++) {
      if (Number(sortedElements[j].id) === Number(sortList[i])) {

        if(i === 0){
          sort_x_pos = 100 + sortCount;
          sort_y_pos = 100 + sortCount;
        } else if(isBlock){
          if ((sort_y_pos + 330) <= 639){
            sort_y_pos += 330;
            console.log('plus 330');
          }else{
            sort_x_pos += 200;
            sort_y_pos = 100 + sortCount;
            console.log('new line');
          }
        } else if (sort_y_pos < 589){
            if(sortedElements[j].sort !== undefined){
               sort_y_pos += 70;
               console.log('589 else');
            }
        } else{
          sort_x_pos += 200;
          sort_y_pos = 100 + sortCount;
          console.log('last else');
        }

        sortedElements[j].position = {
          x: sort_x_pos,
          y: sort_y_pos,
        };

        console.log(sort_x_pos, sort_y_pos);
        console.log(sortedElements[j].position);
        console.log(isBlock);
        console.log(sortedElements[j].sort);

        if ((sortedElements[j].sort !== "0") && (sortedElements[j].sort !== undefined)){
          isBlock = true
        } else{
          isBlock = false;
        }
      }
    }
  }
   setElements(sortedElements);
  console.log(elements)
  sortCount *= -1;
  };


  // 정렬한 노드 list 받아오기
  const sortActive=(event)=>{
    console.log('생성버튼클릭');
    axios.delete("/api/sort/1/")
    .then(function(response){
        console.log(response)
        })
        .catch(e => console.log(e))
    console.log('delete done');
    axios.post("/api/sort/")
        .then(function(response){
        console.log(response)
        axios.get('/api/sort/1/')
            .then(function(response2){
            console.log('정렬된 list: ', response2.data.sorted_ids)
              onSortNodes(response2.data.sorted_ids);

            })
        })
        .catch(e => console.log(e))
    console.log('post done');
  };

  const onLoad = (rFInstance) => setReactFlowInstance(rFInstance);





  //
  if(checkFirst == 0){
      console.log('실행')
      console.log('running_id', running_id)
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

//  const notRunningState = setInterval(()=>{
////    console.log("[post] 동작 중지");
//    running_id += 1;
//    axios.post("/api/status_request/", {
//
//      timestamp: Date.now(),
////      running: 0,
//    }).then(function(response){
//        //console.log(timestamp)
//        })
//        .catch(e => console.log(e));
//    }, initRunningStateTime * 1000)

// const onRunningState = (()=>{
////    console.log("[post] 동작 중");
//   running_id += 1;
//   axios.post("/api/running/", {
//     id : running_id,
//     running: 1,
//   }).then(function(response){
//     console.log(response)
//     })
//     .catch(e => console.log(e));
// })


  const onRunningStateClick = (e) => {
    e.preventDefault();
    //clearInterval(notRunningState);
    //onRunningState();
    //clearInterval(notRunningState);
    //notRunningState();
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
      var maxId = 0;
      for(var i=0; i<cedge.data.length; i++){
       if(maxId<cedge.data[i].id){
        maxId = cedge.data[i].id
       }
      }
      axios.post("/api/edge/",{
        id: maxId+1,
        prior: params.source,
        next: params.target
      }).then(function(response){
        console.log(response)
      }).catch(err=>console.log(err));
  };

  const onDeleteEdge = (e) => {
    console.log(e.target);
  }



  const onElementsRemove = (remove) => {
    setElements((els)=>removeElements(remove, els));
    deleteModal(remove);
  }


  const openModal = async () => {
    // const get_params = async () => {
    //   try {
    //     await axios.get('/api/node/'.concat(String(idState)).concat('/')).then((response) => {
    //        setParam(response.data.parameters);
    //     });
    //   } catch (error) {
    //     console.error(error);
    //   }
    // };
    // await get_params();
    // console.log('get param double click')
    await setModalOpen(true);
    console.log('open modal')
  };


  const closeModal = () => {
    setModalOpen(false);
  };

  const saveModal = () => {
    setModalOpen(false);
  };

  const deleteModal = (remove) => {
    axios.get("/api/node/".concat(String(idState)).concat('/'))
    .then(function(response){
    console.log(response)});
    console.log("remove", remove)
    if(remove[0].data){
        console.log('node')
        axios.delete("/api/node/".concat(String(idState)).concat('/'));
        axios.get("/api/edge/")
        .then(function(response){
        for(var i=0;i<response.data.length;i++){
            if(String(response.data[i].prior) === String(idState)){
                axios.delete("/api/edge/".concat(String(response.data[i].id)).concat('/'));
            }
            if(String(response.data[i].next) === String(idState)){
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

  const onNodeClick = async (event, node) => {
      // if(modalOpen === true) {
      //   const get_params = async () => {
      //     try {
      //       await axios.get('/api/node/'.concat(String(idState)).concat('/')).then((response) => {
      //         setParam(response.data.parameters);
      //       });
      //     } catch (error) {
      //       console.error(error);
      //     }
      //   };
      //   await setState(node.data.label);
      //   await setIdState(node.id);
      //   await get_params();
      //   console.log('get param one click');
      // }
      // else{
        await setState(node.data.label);
        await setIdState(node.id);
        console.log(node.position);

      // }

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
    for (var i = 0; i < cnode.data.length; i++) {
      if (maxOrder < cnode.data[i].order) {
        maxOrder = cnode.data[i].order
      }
    }

    // 가장 큰 order+1로 id값 설정
    const nid = maxOrder + 1;
    id = nid

    //node create **********************
    //const cnode = plusId()
    axios.post("/api/node/", {
      order: id,
      layer: name,
      parameters: subp
    }).then(function (response) {
      console.log(response)
    }).catch(err => console.log(err));
    //node create **********************


    const newNode = {
      id: getId(),
      type: "default",
      position,
      sort: "0",
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

    const newResidualNode1 = {
      // 노드 내부에 residual block 이미지 넣기 - bottleneck
      id: getId(),
      type: "default",
      position,
      sort: "2",
      style: {
        background: `${color}`,
        fontSize: "20px",
        width: "135px",
        height: "280px",
        boxShadow: "7px 7px 7px 0px rgba(0,0,0,.20)",
        border: "0px",
        borderRadius: "10px",
        backgroundImage: `url(${BottleNeckimg})`, //사진 나오게
        backgroundPosition: "center",
        backgroundSize: "135px 280px",
        backgroundRepeat: "no-repeat",
        color: "rgba(0, 0, 0, 0)",
      },
      data: {
        label: `${name}`,
        subparam: `${subp}`
      }
    };
    const newResidualNode2 = {
      // 노드 내부에 residual block 이미지 넣기 - basic block
      id: getId(),
      type: "default",
      position,
      sort: "1",
      style: {
        background: `${color}`,
        fontSize: "20px",
        width: "135px",
        height: "280px",
        boxShadow: "7px 7px 7px 0px rgba(0,0,0,.20)",
        border: "0px",
        borderRadius: "10px",
        backgroundImage: `url(${BasicBlockimg})`, //사진 나오게
        backgroundPosition: "center",
        backgroundSize: "135px 280px",
        backgroundRepeat: "no-repeat",
         color: "rgba(0, 0, 0, 0)",
      },
      data: {
        label: `${name}`,
        subparam: `${subp}`
      }
    };

    if (name == "Bottleneck") {
      setElements((nds) => nds.concat(newResidualNode1));
    } else if (name == "BasicBlock") {
      setElements((nds) => nds.concat(newResidualNode2));
    } else {
      setElements((nds) => nds.concat(newNode));
    }
  };

  const   C = () => {
    if (state === "Conv2d")
      return (
        <EditModal
          params={paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></EditModal>
      );
    if (state === "MaxPool2d")
      return (
        <MaxPoolModal
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></MaxPoolModal>
      );
    if (state === "AvgPool2d")
      return (
        <AvgPool2d
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></AvgPool2d>
      );
    if (state === "AdaptiveAvgPool2d")
      return (
        <AdaptiveAvgPool2d
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></AdaptiveAvgPool2d>
      );
     if (state === "Softmax")
      return (
        <Softmax
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Softmax>
      );
    if (state === "ConstantPad2d")
      return (
        <ConstantPad2d
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></ConstantPad2d>
        );
    if (state === "BatchNorm2d")
      return (
        <BatchNorm2d
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></BatchNorm2d>
      );

    if (state === "MSELoss")
      return (
        <MSELoss
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></MSELoss>
      );
    if (state === "Tanh")
      return (
        <Tanh
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Tanh>
      );
    if (state === "Sigmoid")
      return (
        <Sigmoid
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Sigmoid>
      );
    if (state === "CrossEntropyLoss")
      return (
        <CrossEntropyLoss
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></CrossEntropyLoss>
      );
    if (state === "Linear")
      return (
        <Linear
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Linear>
      );
    if (state === "Dropout")
      return (
        <Dropout
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Dropout>
      );
      if (state === "ZeroPad2d")
      return (
        <ZeroPad2d
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></ZeroPad2d>
      );
      if (state === "BCELoss")
      return (
        <BCELoss
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></BCELoss>
      );
      if (state === "LeakyReLU")
      return (
        <LeakyReLU
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></LeakyReLU>
      );
       if (state === "ReLU")
      return (
        <ReLU
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></ReLU>
      );
      if (state === "ReLU6")
      return (
        <ReLU6
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></ReLU6>
      );
       if (state === "Flatten")
      return (
        <Flatten
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Flatten>
      );
       if (state === "BasicBlock")
      return (
        <BasicBlock
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></BasicBlock>
      );
       if (state === "Bottleneck")
      return (
        <Bottleneck
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Bottleneck>
      );

    else
      return (
        <Upsample
          params = {paramState}
          layer={idState}
          open={modalOpen}
          save={saveModal}
          close={closeModal}
          header={state}
          setState={setIdState}
        ></Upsample>
      );
  };

  const [tabToggle, setTabtoggle] = useState(1)
const tabOnClick = (path) => {
  console.log(path)
  if (path == 'info icon') {
    setTabtoggle(2)
  } else {
    setTabtoggle(1)
  }

}

// const onSortNodes = () => {
//     console.log('back code');
//     sortActive();
//     console.log(sortList);
//   console.log(' my code ');
//   const sortList = [1, 2, 288, 4, 5, 6, 7, 8, 9]; // 정렬된 노드 ID 리스트
//   const sortedElements = elements.slice(); // elements 배열을 복사하여 새로운 배열을 생성합니다.
//   let sort_x_pos = 100 + sortCount;
//   let sort_y_pos = 100 + sortCount;
//
//
//   for(var i = 0; i < sortList.length; i++) {
//     for (var j = 0; j < sortedElements.length; j++) {
//       if (sortedElements[j].id === sortList[i]) {
//         console.log('arrange');
//         console.log(sortList[i]);
//         // node = sortedElements[j];
//
//         if ((i % 8 === 0) && (i >= 8)){
//           sort_x_pos += 200;
//           sort_y_pos = 100;
//         } else if (i>=1) {
//           sort_y_pos += 70;
//         };
//
//         sortedElements[j].position = {
//           x: sort_x_pos,
//           y: sort_y_pos,
//         };
//
//         console.log(sort_x_pos, sort_y_pos);
//         console.log(sortedElements[j].position)
//       }
//     }
//   }
//    setElements(sortedElements);
//   console.log(elements)
//   sortCount *= -1;
//   };


  return (
      <div className="FullPage">
        <div className="Sidebar">
          <Tab tabOnClick={tabOnClick}/>
          {(tabToggle === 1)?<LayerToggle />:<NetworkInformation />}
          {/*<LayerToggle/>*/}
          <div className="LayerInfo">
            <h3>Layer Information</h3>
            {/*<div className="Modal">*/}
              <C />
            </div>
         </div>


    <div className="dndflow" >
      <ReactFlowProvider>
        <div className="reactflow-wrapper" ref={reactFlowWrapper}>
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
            onElementsRemove={onElementsRemove}
            onElementClick={onNodeClick}

          >
            <Controls showZoom="" showInteractive="" showFitView="">
              {/*정렬된 노드 get 요청*/}
              <ControlButton onClick={sortActive} title="action">
                <img src={arange_icon}/>
              </ControlButton>
            </Controls>
          <div className="reactBtn" style={{position:'absolute' ,zIndex:100}}>

            <GenerateButton  elements={elements}  />
            </div>

          </ReactFlow>
        </div>
        </ReactFlowProvider>

    </div>
      </div>
  );
}

export default function Layer() {
  return <LayerList />;
}
