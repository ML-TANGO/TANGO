import axios from 'axios';

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
    for (var j=0; j<60; j++){

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
    for (var j=0; j<60; j++){

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

  const jsonData = require('./VGG16.json');

  var node_id = 1;
  var edge_id = 1;
  var x_pos = 100;
  var y_pos = 100;


  var initElements = [];

  for (var i = 0; i < jsonData.node.length; i++) {
    let nodeOrder = jsonData.node[i].order;
    let nodeLabel = jsonData.node[i].layer;
    let nodeParam = jsonData.node[i].parameters;

    const newNode = {
      id: String(nodeOrder),
      type: "default",
      position: { x: x_pos, y: y_pos },
      style: {
        background: "#434854",
        width: 150,
        color: "#fff",
        fontSize: "20px",
        fontFamily: "Helvetica",
        boxShadow: "5px 5px 5px 0px rgba(0,0,0,.10)"
      },
      data: {
        // index: `${nodeOrder}`,
        label: `${nodeLabel}`,
        subparam: `${nodeParam}`
      }

      
    };

    // post the new node
    axios.post("/api/node/",{
      order: String(nodeOrder),
      layer: nodeLabel,
      parameters: nodeParam
    }).then(function(response){
      console.log(response)
    });

    initElements.push(newNode);
    // node_id += 1;
    if (i % 8 == 7) {
      x_pos += 200;
      y_pos = 100;
    } else {
      y_pos += 70;
    }
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
