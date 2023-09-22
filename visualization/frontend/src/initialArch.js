import axios from 'axios';
import NodeColorProp from "./NodeColor";
import BottleNeckimg from "./img/bottleneck.png";
import BasicBlockimg from "./img/basicblock.png";

export const initialArch = () => {
  var checkFirst = 0;

  if (checkFirst==0){
//    axios.post("/api/start/", {
//      start: 0,
//    }).then(function(response){
//        console.log(response)
//        })
//        .catch(e => console.log(e));

      // 노드와 엣지 삭제하기
    for (var j=0; j<300; j++){

    axios.delete('/api/node/'.concat(j).concat('/'))
     .then(function (response) {
       // handle success
     })
     .catch(function (error) {
       // handle error
     })
     .then(function () {
       // always executed
     });
    }

    //노드와 엣지 삭제하기
    for (var j=0; j<300; j++){

    axios.delete('/api/edge/'.concat(j).concat('/'))
     .then(function (response) {
       // handle success
     })
     .catch(function (error) {
       // handle error
     })
     .then(function () {
       // always executed
     });
    }

  const jsonData = require('./resnet50.json');

  var node_id = 1;
  var edge_id = 1;
  var x_pos = 100;
  var y_pos = 100;
  var isBlock = undefined;

  var initElements = [];

  for (var i = 0; i < jsonData.node.length; i++) {
    let nodeOrder = jsonData.node[i].order;
    let nodeLabel = jsonData.node[i].layer;
    let nodeParam = jsonData.node[i].parameters;
    let nodeColor ;
    if (nodeLabel === "Conv2d"){
      nodeColor = NodeColorProp.Conv;
    }
    else if (nodeLabel === "MaxPool2d"){
       nodeColor = NodeColorProp.Pooling;
    }
    else if (nodeLabel === "AvgPool2d"){
       nodeColor = NodeColorProp.Pooling;
    }
    else if (nodeLabel === "AdaptiveAvgPool2d"){
       nodeColor = NodeColorProp.Pooling;
    }
    else if(nodeLabel === "ZeroPad2d"){
       nodeColor = NodeColorProp.Padding;
    }
    else if(nodeLabel === "ConstantPad2d"){
       nodeColor = NodeColorProp.Padding;
    }
    else if(nodeLabel === "ReLU"){
       nodeColor = NodeColorProp.Activation;
    }
    else if(nodeLabel  === "ReLU6"){
       nodeColor = NodeColorProp.Activation;
    }
    else if(nodeLabel === "Sigmoid"){
       nodeColor = NodeColorProp.Activation;
    }
    else if(nodeLabel === "LeakyReLU"){
       nodeColor = NodeColorProp.Activation;
    }
    else if(nodeLabel === "Tanh"){
       nodeColor = NodeColorProp.Activation;
    }else if(nodeLabel === "Softmax"){
       nodeColor = NodeColorProp.Activation;
    }
    else if(nodeLabel ===  "BatchNorm2d"){
       nodeColor = NodeColorProp.Normalization;
    }
    else if(nodeLabel === "Linear"){
       nodeColor = NodeColorProp.Linear;
    }
    else if(nodeLabel === "Dropout"){
       nodeColor = NodeColorProp.Dropout;
    }
    else if(nodeLabel === "BCELoss"){
       nodeColor = NodeColorProp.Loss;
    }
    else if(nodeLabel === "CrossEntropyLoss"){
       nodeColor = NodeColorProp.Loss;
    }
    else if(nodeLabel === "Flatten"){
       nodeColor = NodeColorProp.Utilities;
    }
    else if(nodeLabel === "Upsample"){
       nodeColor = NodeColorProp.Vision;
    }
    else  if(nodeLabel === "MSELoss"){
        nodeColor = NodeColorProp.Loss;
    }else if(nodeLabel === "BasicBlock"){
      nodeColor = NodeColorProp.Residual;
    }else if(nodeLabel === "Bottleneck"){
      nodeColor = NodeColorProp.Residual;
    }

    if(i === 0){
      x_pos = 100;
      y_pos = 100;
    } else if(isBlock){
      if((y_pos + 330) <= 639){
        y_pos += 330;
      }else{
        x_pos += 200;
        y_pos = 100;
      }
    } else if(y_pos < 589){
      y_pos += 70;
    } else {
      x_pos += 200;
      y_pos = 100;
    }

    if((String(nodeLabel) === 'BasicBlock') || (String(nodeLabel) === 'Bottleneck')){
      isBlock = true;
    }else{
      isBlock = false;
    }

    const newNode = {
      id: String(nodeOrder),
      type: "default",
      position: { x: x_pos, y: y_pos },
      sort: "0",
      style: {
        background: `${nodeColor}`,
        width: 135,
        color: "black",
        fontSize: "20px",
        fontFamily: "Helvetica",
        boxShadow: "5px 5px 5px 0px rgba(0,0,0,.10)",
        borderRadius: "10px",
        border: "none"
      },
      data: {
        // index: `${nodeOrder}`,
        label: `${nodeLabel}`,
        subparam: `${nodeParam}`
      }
    };

    const newResidualNode1 = {
      // 노드 내부에 residual block 이미지 넣기 - bottleneck
      id: String(nodeOrder),
      type: "default",
      position:{x: x_pos, y:y_pos},
      sort: "2",
      style: {
        background: `${nodeColor}`,
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
        label: `${nodeLabel}`,
        subparam: `${nodeParam}`
      }
    };
    const newResidualNode2 = {
      // 노드 내부에 residual block 이미지 넣기 - basic block
      id: String(nodeOrder),
      type: "default",
      position:{x: x_pos, y:y_pos},
      sort: "1",
      style: {
        background: `${nodeColor}`,
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
        label: `${nodeLabel}`,
        subparam: `${nodeParam}`
      }
    };

     //post the new node
    axios.post("/api/node/",{
      order: String(nodeOrder),
      layer: nodeLabel,
      parameters: nodeParam
    }).then(function(response){
      console.log(response)
    });
    if(String(nodeLabel) === 'Bottleneck'){
      initElements.push(newResidualNode1);
    } else if(String(nodeLabel) === 'BasicBlock'){
      initElements.push(newResidualNode2);
    } else{
      initElements.push(newNode);
    }

    // node_id += 1;
    // if ((i % 8 === 7) || (y_pos  > 430)) {
    //   x_pos += 200;
    //   y_pos = 100;
    // } else if((String(nodeLabel) === 'BasicBlock') || (String(nodeLabel) === 'Bottleneck')){
    //   y_pos += 330;
    // } else {
    //   y_pos += 70;
    // }
    node_id += 1;
  }

  for (var j = 0; j < jsonData.edge.length; j++) {
    let edgeId = jsonData.edge[j].id;
    let edgeNext = jsonData.edge[j].next;
    let edgePrior = jsonData.edge[j].prior;

    const newEdge = {
      id: String(edgeId + node_id),
      // id: "reactflow__edge-"+ `${edgePrior}` + "null" + `${edgeNext}` + "null",
      source: String(edgeNext),
      sourceHandle: null,
      target: String(edgePrior),
      targetHandle: null
    }

    // post the new edge
    axios.post("/api/edge/",{
      id: String(edgeId),
      prior: String(edgePrior),
      next: String(edgeNext)
    }).then(function(response){
      console.log(response)
    });

    initElements.push(newEdge);
    // _id = _id + 1;
  }
    checkFirst = 1;

  }



  console.log("[initElements]", initElements)

  return initElements;
}
