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
    const [info, setInfo] = useState("");
    const [data, setData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [checkFirst, setCheckFirst] = useState(level);

    useEffect(() => {
        setIsLoading(true);

        const init = async () => {
            function renderData(cnode, dedge) {
                var node_id = 1;
                var edge_id = 1;
                var x_pos = 100;
                var y_pos = 100;
                var isBlock = undefined;
                //그 배열을 화면에 보여줌
                var initElements = [];
                // var GNodeIdList = [];
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
                            border: "none",
                        },
                        data: {
                            // index: `${nodeOrder}`,
                            label: `${nodeLabel}`,
                            subparam: `${nodeParam}`,
                        },
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
                            subparam: `${nodeParam}`,
                        },
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
                            subparam: `${nodeParam}`,
                        },
                    };


                    // GNodeIdList.push(node_id);
                    if (String(nodeLabel) === "Bottleneck") {
                        initElements.push(newResidualNode1);
                        // node_id++;
                    } else if (String(nodeLabel) === "BasicBlock") {
                        initElements.push(newResidualNode2);
                        // node_id++;
                    } else {
                        initElements.push(newNode);
                        // node_id++;
                    }

                    node_id++;
                }
                //    edge 설정
                // console.log(GNodeIdList);
                for (var j = 0; j < dedge.data.length; j++) {
                    let edgeId = dedge.data[j].id;
                    let edgeNext = dedge.data[j].next;
                    let edgePrior = dedge.data[j].prior;
                    const newEdge = {
                        id: String(edge_id + node_id),
                        // id: "reactflow__edge-"+ `${edgePrior}` + "null" + `${edgeNext}` + "null",
                        source: String(edgeNext),
                        sourceHandle: null,
                        target: String(edgePrior),
                        targetHandle: null
                    };
                    initElements.push(newEdge);
                    // edge_id++;
                }
                //
                console.log("initElements", initElements);
                setData([...initElements]);
                setIsLoading(false);
                setIsSort(false);
                setCheckFirst(checkFirst+1);
            }

            if (checkFirst === 0) {
                console.log("맨 처음 실행코드-");
                // get general information -------------------------------------
                const get_info = async () => {
                    try {
                        return await axios.get("/api/info");
                    } catch (error) {
                        console.error(error);
                    }
                };
                const info_list = await get_info();
                const k = info_list.data.length
                console.log(k)
                const info = info_list.data[k-1]
                setInfo(info);

                const info_id = info.id
                const userid = info.userid
                const project_id = info.project_id
                const model = info.model_type + info.model_size
                const status = info.model_viz
                console.log('id=', info_id, 'userid=', userid, ' project_id=', project_id)
                console.log('model=', model, 'ready to load model=', status)

                // get nodes and edges -----------------------------------------
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

                const cnode = await get_node();
                const dedge = await get_edge();

                // render nodes and edges --------------------------------------
                renderData(cnode, dedge);

                // report django to finish rendering model ---------------------
                const patch_data = {'model_viz': 'done'}
                axios.patch("/api/info/".concat(String(info.id)).concat('/'), patch_data)
                    .then((response) => {
                        console.log(response.data)
                    })
                    .catch((error) => {
                        console.error(error)
                    });
            }
            else {
                console.log("level1 두번째부터 실행하는 코드");
                // get nodes and edges -----------------------------------------
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
                const cnode = await get_node();
                const dedge = await get_edge();

                // render nodes and edges --------------------------------------
                renderData(cnode, dedge);
            }
        };
        init();
    }, [level, group, setGroup, ungroup, setUngroup, isSort, setIsSort]);


    return [info, data, setData, isLoading];
}
export default InitialArch;
