import React, {
    useEffect,
    useState
} from "react";
import axios from 'axios';
import NodeColorProp from "./NodeColor";
import BottleNeckimg from "./img/bottleneck.png";
import BasicBlockimg from "./img/basicblock.png";
import Layer from "./components/page/Layer"


function InitialArch(level, group, setGroup, ungroup, setUngroup, isSort, setIsSort) {
    const [data, setData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [checkFirst, setCheckFirst] = useState(0);


    useEffect(() => {
        setIsLoading(true);
        console.log("useInitialArch useEffect");
        console.log("group", group);
        console.log("ungroup", ungroup);
        console.log("isSort", isSort);
        const init = async () => {
            function renderData(resData) {
                // node_id 와 edge_id로 json 파일을 읽어 순서대로 새로운 id 를 부여함
                var node_id = 1;
                var edge_id = 1;
                var x_pos = 100;
                var y_pos = 100;
                var isBlock = undefined;
                //그 배열을 화면에 보여줌
                var initElements = [];
                var GNodeIdList = [];
                // json 파일에서 파일 output의 길이만큼 읽어옴
                for (var i = 0; i < resData.data.output.length; i++) {
                    let nodeLabel = resData.data.output[i].layer;
                    let nodeId = resData.data.output[i].nodeId;
                    let parameters = resData.data.output[i].parameters;
                    let nodeColor;
                    if (nodeLabel === "Conv2d") {
                        nodeColor = NodeColorProp.Conv;
                    } else if (nodeLabel === "Conv") {
                        nodeColor = NodeColorProp.Conv
                    } else if (nodeLabel === "MaxPool2d") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "AvgPool2d") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "AdaptiveAvgPool2d") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "MP") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "SP") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "ZeroPad2d") {
                        nodeColor = NodeColorProp.Padding;
                    } else if (nodeLabel === "ConstantPad2d") {
                        nodeColor = NodeColorProp.Padding;
                    } else if (nodeLabel === "ReLU") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "ReLU6") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "Sigmoid") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "LeakyReLU") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "Tanh") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "Softmax") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "BatchNorm2d") {
                        nodeColor = NodeColorProp.Normalization;
                    } else if (nodeLabel === "Linear") {
                        nodeColor = NodeColorProp.Linear;
                    } else if (nodeLabel === "Dropout") {
                        nodeColor = NodeColorProp.Dropout;
                    } else if (nodeLabel === "BCELoss") {
                        nodeColor = NodeColorProp.Loss;
                    } else if (nodeLabel === "CrossEntropyLoss") {
                        nodeColor = NodeColorProp.Loss;
                    } else if (nodeLabel === "Flatten") {
                        nodeColor = NodeColorProp.Utilities;
                    } else if (nodeLabel === "Upsample") {
                        nodeColor = NodeColorProp.Vision;
                    } else if (nodeLabel === "MSELoss") {
                        nodeColor = NodeColorProp.Loss;
                    } else if (nodeLabel === "BasicBlock") {
                        nodeColor = NodeColorProp.Residual;
                    } else if (nodeLabel === "Bottleneck") {
                        nodeColor = NodeColorProp.Residual;
                    } else if (nodeLabel === "Concat") {
                        nodeColor = NodeColorProp.Concat;
                    } else if (nodeLabel === "Shortcut") {
                        nodeColor = NodeColorProp.Sum;
                    } else if (nodeLabel === "DownC") {
                        nodeColor = NodeColorProp.SPP;
                    } else if (nodeLabel === "SPPCSPC") {
                        nodeColor = NodeColorProp.SPP;
                    } else if (nodeLabel === "ReOrg") {
                        nodeColor = NodeColorProp.Utilities
                    } else if (nodeLabel === "IDetect") {
                        nodeColor = NodeColorProp.Head
                    }

                    if (i === 0) {
                        x_pos = 100;
                        y_pos = 100;
                    } else if (isBlock) {
                        if (y_pos + 330 <= 639) {
                            y_pos += 330;
                        } else {
                            x_pos += 200;
                            y_pos = 100;
                        }
                    } else if (y_pos < 589) {
                        y_pos += 70;
                    } else {
                        x_pos += 200;
                        y_pos = 100;
                    }

                    if (
                        String(nodeLabel) === "BasicBlock" ||
                        String(nodeLabel) === "Bottleneck"
                    ) {
                        isBlock = true;
                    } else {
                        isBlock = false;
                    }

                    const newNode = {
                        id: String(node_id),
                        type: "default",
                        position: {
                            x: x_pos,
                            y: y_pos
                        },
                        sort: "0",
                        style: {
                            background: `${nodeColor}`,
                            width: 135,
                            color: "black",
                            fontSize: "20px",
                            fontFamily: "Helvetica",
                            boxShadow: "5px 5px 5px 0px rgba(0,0,0,.10)",
                            borderRadius: "10px",
                            border: "none",
                        },
                        data: {
                            // index: `${nodeOrder}`,
                            label: `${nodeLabel}`,
                            // subparam: `${nodeParam}`,
                        },
                    };

                    const newResidualNode1 = {
                        // 노드 내부에 residual block 이미지 넣기 - bottleneck
                        id: String(node_id),
                        type: "default",
                        position: {
                            x: x_pos,
                            y: y_pos
                        },
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
                            // subparam: `${nodeParam}`,
                        },
                    };

                    const newResidualNode2 = {
                        // 노드 내부에 residual block 이미지 넣기 - basic block
                        id: String(node_id),
                        type: "default",
                        position: {
                            x: x_pos,
                            y: y_pos
                        },
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
                            // subparam: `${nodeParam}`,
                        },
                    };


                    GNodeIdList.push(node_id);
                    if (String(nodeLabel) === "Bottleneck") {
                        initElements.push(newResidualNode1);
                        node_id++;
                    } else if (String(nodeLabel) === "BasicBlock") {
                        initElements.push(newResidualNode2);
                        node_id++;
                    } else {
                        initElements.push(newNode);
                        node_id++;
                    }
                }
                //    edge 설정
                console.log(GNodeIdList);
                for (var j = 0; j < resData.data.output.length; j++) {
                    const newEdge = {
                        id: String(edge_id),
                        // id: "reactflow__edge-"+ `${edgePrior}` + "null" + `${edgeNext}` + "null",
                        source: String(GNodeIdList[j]),
                        sourceHandle: null,
                        target: String(GNodeIdList[j + 1]),
                        targetHandle: null,
                    };
                    initElements.push(newEdge);
                    edge_id++;
                }
                //
                console.log("initElements", initElements);
                setData([...initElements]);
                setIsLoading(false);

            }
            if (checkFirst === 0) {
                console.log("맨 처음 실행코드-");
//                for (var j = 0; j < 160; j++) {
//                    axios.delete('/api/node/'.concat(j).concat('/'))
//                        .then(function(response) {})
//                        .catch(function(error) {})
//                        .then(function() {});
//                }
//                for (var j = 0; j < 150; j++) {
//                    axios.delete('/api/edge/'.concat(j).concat('/'))
//                        .then(function(response) {})
//                        .catch(function(error) {})
//                        .then(function() {});
//                }
                //            const response10 = await axios.post("/start", {
                //                msg: 'started',
                //                user_id: 123,
                //                project_id: 123
                //            }).then(async function(response) {
                const get_node = async () => {
                          try {
                            return await axios.get("/api/node/");
                          } catch (error) {
                            console.error(error);
                          }
                  };

                const get_edge = async () => {
                          try {
                            return await axios.get("/api/edge/");
                          } catch (error) {
                            console.error(error);
                          }
                };
                // console.log("sadstjklsdfhludz")
                const cnode = await get_node();
                const dedge = await get_edge();
                var node_id = 1;
                var edge_id = 1;
                var x_pos = 100;
                var y_pos = 100;
                var isBlock = undefined;
                var initElements = [];

                for (var i = 0; i < cnode.data.length; i++) {
                    let nodeOrder = cnode.data[i].order;
                    let nodeLabel = cnode.data[i].layer;
                    let nodeParam = cnode.data[i].parameters;
                    let nodeColor;
                    if (nodeLabel === "Conv2d") {
                        nodeColor = NodeColorProp.Conv;
                    } else if (nodeLabel === "Conv") {
                        nodeColor = NodeColorProp.Conv
                    } else if (nodeLabel === "MaxPool2d") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "AvgPool2d") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "AdaptiveAvgPool2d") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel == "MP") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel == "SP") {
                        nodeColor = NodeColorProp.Pooling;
                    } else if (nodeLabel === "ZeroPad2d") {
                        nodeColor = NodeColorProp.Padding;
                    } else if (nodeLabel === "ConstantPad2d") {
                        nodeColor = NodeColorProp.Padding;
                    } else if (nodeLabel === "ReLU") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "ReLU6") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "Sigmoid") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "LeakyReLU") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "Tanh") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "Softmax") {
                        nodeColor = NodeColorProp.Activation;
                    } else if (nodeLabel === "BatchNorm2d") {
                        nodeColor = NodeColorProp.Normalization;
                    } else if (nodeLabel === "Linear") {
                        nodeColor = NodeColorProp.Linear;
                    } else if (nodeLabel === "Dropout") {
                        nodeColor = NodeColorProp.Dropout;
                    } else if (nodeLabel === "BCELoss") {
                        nodeColor = NodeColorProp.Loss;
                    } else if (nodeLabel === "CrossEntropyLoss") {
                        nodeColor = NodeColorProp.Loss;
                    } else if (nodeLabel === "Flatten") {
                        nodeColor = NodeColorProp.Utilities;
                    } else if (nodeLabel === "Upsample") {
                        nodeColor = NodeColorProp.Vision;
                    } else if (nodeLabel === "MSELoss") {
                        nodeColor = NodeColorProp.Loss;
                    } else if (nodeLabel === "BasicBlock") {
                        nodeColor = NodeColorProp.Residual;
                    } else if (nodeLabel === "Bottleneck") {
                        nodeColor = NodeColorProp.Residual;
                    } else if (nodeLabel === "Concat") {
                        nodeColor = NodeColorProp.Concat;
                    } else if (nodeLabel === "Shortcut") {
                        nodeColor = NodeColorProp.Sum;
                    } else if (nodeLabel === "DownC") {
                        nodeColor = NodeColorProp.SPP;
                    } else if (nodeLabel === "ReOrg") {
                        nodeColor = NodeColorProp.Utilities
                    } else if (nodeLabel === 'IDetect') {
                        nodeColor = NodeColorProp.Head
                    }
                    // console.log("sadstjklsdfhludz")
                    if (i === 0) {
                        x_pos = 100;
                        y_pos = 100;
                    } else if (isBlock) {
                        if ((y_pos + 330) <= 639) {
                            y_pos += 330;
                        } else {
                            x_pos += 200;
                            y_pos = 100;
                        }
                    } else if (y_pos < 589) {
                        y_pos += 70;
                    } else {
                        x_pos += 200;
                        y_pos = 100;
                    }
                    // console.log("sadstjklsdfhludz")
                    if ((String(nodeLabel) === 'BasicBlock') || (String(nodeLabel) === 'Bottleneck')) {
                        isBlock = true;
                    } else {
                        isBlock = false;
                    }
                    // console.log("sadstjklsdfhludz")
                    const newNode = {
                        id: String(nodeOrder),
                        type: "default",
                        position: {
                            x: x_pos,
                            y: y_pos
                        },
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
                        position: {
                            x: x_pos,
                            y: y_pos
                        },
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
                        position: {
                            x: x_pos,
                            y: y_pos
                        },
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
//                    axios.post("/api/node/", {
//                        order: String(nodeOrder),
//                        layer: nodeLabel,
//                        parameters: nodeParam
//                    }).then(function(response) {
//                        console.log(response)
//                    });
                    if (String(nodeLabel) === 'Bottleneck') {
                        initElements.push(newResidualNode1);
                    } else if (String(nodeLabel) === 'BasicBlock') {
                        initElements.push(newResidualNode2);
                    } else {
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

                for (var j = 0; j < dedge.data.length; j++) {
                    let edgeId = dedge.data[j].id;
                    let edgeNext = dedge.data[j].next;
                    let edgePrior = dedge.data[j].prior;

                    const newEdge = {
                        id: String(edgeId + node_id),
                        // id: "reactflow__edge-"+ `${edgePrior}` + "null" + `${edgeNext}` + "null",
                        source: String(edgeNext),
                        sourceHandle: null,
                        target: String(edgePrior),
                        targetHandle: null
                    };

                    // post the new edge
//                    axios.post("/api/edge/", {
//                        id: String(edgeId),
//                        prior: String(edgePrior),
//                        next: String(edgeNext)
//                    }).then(function(response) {
//                        console.log(response)
//                    });

                    initElements.push(newEdge);
                    // console.log("sadstjklsdfhludz")
                }
                // _id = _id + 1;

                console.log(initElements);
                setData([...initElements]);
                setIsLoading(false);
                setIsSort(false);
                setCheckFirst(1);
            }
            else {
                console.log("level1 두번째부터 실행하는 코드");
                axios.get("/api/node/").then(function(response2){
                    renderData(response2);
                })
                setCheckFirst(1);
            }
        };
        init();
    }, [level, group, setGroup, ungroup, setUngroup, isSort, setIsSort]);


    return [data, setData, isLoading];
}
export default InitialArch;
