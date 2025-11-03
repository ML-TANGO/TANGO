import React, { useEffect, useState } from "react";
import axios from 'axios';
import NodeColorProp from "./NodeColor";
import BottleNeckimg from "./img/bottleneck.png";
import BasicBlockimg from "./img/basicblock.png";

function useInitialArch(level, group, ungroup, isSort, isYolo = false) {  // isYolo만 추가, setter 제거
    const [data, setData] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        console.log('[useInitialArch] recompute, isYolo =', isYolo)
        let cancelled = false
        setIsLoading(true);

        (async () => {
            try {
                // 서버에서 노드와 엣지 데이터를 가져옴 (삭제하지 않음)
                const [nodeResponse, edgeResponse] = await Promise.all([
                    axios.get("/api/node/"),
                    axios.get("/api/edge/")
                ]);

                if (cancelled) return;

                const resData = { nodes: nodeResponse.data, edges: edgeResponse.data };
                const computedData = renderData(resData, isYolo)
                if (!cancelled) {
                    setData(computedData);
                    setIsLoading(false);
                }
            } catch (error) {
                if (!cancelled) setIsLoading(false);
                console.error("노드 또는 엣지 가져오는 중 오류:", error);
            }
        })();

        return () => { cancelled = true; };
    }, [level, group, ungroup, isSort, isYolo]);

    return [data, setData, isLoading];
}

function renderData(resData, isYolo) {
    const nodeOrderToIdMap = {};
    const initElements = [];
    // var GNodeIdList = [];
    let node_id = 1;
    let x_pos = 100;
    let y_pos = 100;
    let isBlock = undefined;

    // 노드를 order 기준으로 정렬
    const sortedNodes = resData.nodes.sort((a, b) => a.order - b.order);
    sortedNodes.forEach((node) => {
        nodeOrderToIdMap[node.order] = node_id;

        // nodeColor 계산: isYolo 여부에 따른 노드 색상 로직
        let nodeColor;
        if (!isYolo) {
            switch (node.layer) {
                case "Conv2d":
                case "Conv":
                    nodeColor = NodeColorProp.Conv;
                    break;
                case "MaxPool2d":
                case "AvgPool2d":
                case "AdaptiveAvgPool2d":
                    nodeColor = NodeColorProp.Pooling;
                    break;
                case "ZeroPad2d":
                case "ConstantPad2d":
                    nodeColor = NodeColorProp.Padding;
                    break;
                case "ReLU":
                case "ReLU6":
                case "Sigmoid":
                case "LeakyReLU":
                case "Tanh":
                case "Softmax":
                    nodeColor = NodeColorProp.Activation;
                    break;
                case "BatchNorm2d":
                    nodeColor = NodeColorProp.Normalization;
                    break;
                case "Linear":
                    nodeColor = NodeColorProp.Linear;
                    break;
                case "Dropout":
                    nodeColor = NodeColorProp.Dropout;
                    break;
                case "BCELoss":
                case "CrossEntropyLoss":
                case "MSELoss":
                    nodeColor = NodeColorProp.Loss;
                    break;
                case "Flatten":
                case "ReOrg":
                    nodeColor = NodeColorProp.Utilities;
                    break;
                case "Upsample":
                    nodeColor = NodeColorProp.Vision;
                    break;
                case "BasicBlock":
                case "Bottleneck":
                    nodeColor = NodeColorProp.Residual;
                    break;
                case "Concat":
                    nodeColor = NodeColorProp.Concat;
                    break;
                case "Shortcut":
                    nodeColor = NodeColorProp.Sum;
                    break;
                case "DownC":
                case "SPPCSPC":
                    nodeColor = NodeColorProp.SPP;
                    break;
                case "IDetect":
                    nodeColor = NodeColorProp.Head;
                    break;
                default:
                    nodeColor = NodeColorProp.Default;
            }
        } else if (isYolo) {
            switch (node.layer) {
                case "Conv":
                    nodeColor = NodeColorProp.Yolo_Conv;
                    break;
                case "RepNCSPELAN4":
                    nodeColor = NodeColorProp.Yolo_RepNCSPELAN4;
                    break;
                case "ADown":
                    nodeColor = NodeColorProp.Yolo_ADown;
                    break;
                case "Concat":
                    nodeColor = NodeColorProp.Yolo_Concat;
                    break;
                case "Upsample":
                    nodeColor = NodeColorProp.Yolo_Upsample;
                    break;
                case "SPPELAN":
                    nodeColor = NodeColorProp.Yolo_SPPELAN;
                    break;
                case "Detect":
                    nodeColor = NodeColorProp.Yolo_Detect;
                    break;
                default:
                    nodeColor = NodeColorProp.Default;
            }
        }

        // position 계산: isYolo 여부에 따른 좌표 로직
        if (!isYolo) {
            // 노드 위치 설정
            if (node_id === 1) {
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
        } else if (isYolo) {
            switch (node_id) {
                case 1:
                    x_pos = 100;
                    y_pos = 0;
                    break;
                case 2:
                    x_pos = 100;
                    y_pos = 100;
                    break;
                case 3:
                    x_pos = 100;
                    y_pos = 200;
                    break;
                case 4:
                    x_pos = 100;
                    y_pos = 300;
                    break;
                case 5:
                    x_pos = 100;
                    y_pos = 400;
                    break;
                case 6:
                    x_pos = 100;
                    y_pos = 600;
                    break;
                case 7:
                    x_pos = 100;
                    y_pos = 700;
                    break;
                case 8:
                    x_pos = 100;
                    y_pos = 800;
                    break;
                case 9:
                    x_pos = 100;
                    y_pos = 900;
                    break;

                case 10:
                    x_pos = 400;
                    y_pos = 900;
                    break;
                case 11:
                    x_pos = 400;
                    y_pos = 800;
                    break;
                case 12:
                    x_pos = 400;
                    y_pos = 700;
                    break;
                case 13:
                    x_pos = 400;
                    y_pos = 600;
                    break;
                case 14:
                    x_pos = 400;
                    y_pos = 500;
                    break;
                case 15:
                    x_pos = 400;
                    y_pos = 400;
                    break;
                case 16:
                    x_pos = 400;
                    y_pos = 300;
                    break;

                case 17:
                    x_pos = 700;
                    y_pos = 500;
                    break;
                case 18:
                    x_pos = 700;
                    y_pos = 600;
                    break;
                case 19:
                    x_pos = 700;
                    y_pos = 700;
                    break;
                case 20:
                    x_pos = 700;
                    y_pos = 800;
                    break;
                case 21:
                    x_pos = 700;
                    y_pos = 900;
                    break;
                case 22:
                    x_pos = 700;
                    y_pos = 1000;
                    break;

                case 23:
                    x_pos = 1000;
                    y_pos = 300;
                    break;
                case 24:
                    x_pos = 1000;
                    y_pos = 700;
                    break;
                case 25:
                    x_pos = 1000;
                    y_pos = 1000;
                    break;
                default:
                    break;
            }
        }

        const newNode = {
            id: String(node_id),
            type: "custom",
            position: { x: x_pos, y: y_pos },
            sort: "0",
            style: {
                background: nodeColor,
                color: "black",
                fontSize: "20px",
                fontFamily: "Helvetica",
                boxShadow: "5px 5px 5px 0px rgba(0,0,0,.10)",
                borderRadius: "10px",
                border: "none",
            },
            data: {
                label: node.layer,
            },
        };

        // Residual Blocks 처리
        isBlock = (node.layer === "BasicBlock" || node.layer === "Bottleneck");

        if (node.layer === "Bottleneck") {
            newNode.type = "default";
            newNode.style.backgroundImage = `url(${BottleNeckimg})`;
            newNode.style.height = "280px";
            newNode.style.backgroundPosition = "center";
            newNode.style.backgroundSize = "135px 280px";
            newNode.style.backgroundRepeat = "no-repeat";
            newNode.style.color = "rgba(0, 0, 0, 0)";
        } else if (node.layer === "BasicBlock") {
            newNode.type = "default";
            newNode.style.backgroundImage = `url(${BasicBlockimg})`;
            newNode.style.height = "280px";
            newNode.style.backgroundPosition = "center";
            newNode.style.backgroundSize = "135px 280px";
            newNode.style.backgroundRepeat = "no-repeat";
            newNode.style.color = "rgba(0, 0, 0, 0)";
        }

        // GNodeIdList.push(node_id);
        initElements.push(newNode);
        node_id++;
    });

    // --- YOLOv9 전용 노드 ---
    const additionalNodes = isYolo ? [
        {
            id: 'node-1',
            type: 'default',
            position: { x: 80, y: -15 },
            data: {},
            style: {
                width: '195px',
                height: '960px',
                backgroundColor: 'transparent', // '#f4e9f0'
                border: '3px dashed #C00000',
                borderRadius: '10px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '20px',
                color: 'black',
            },
        },
        {
            id: 'node-2',
            type: 'default',
            position: { x: 380, y: 280 },
            data: {},
            style: {
                width: '500px',
                height: '760px',
                backgroundColor: 'transparent', // '#e2ece9'
                border: '3px dashed #81BF97',
                borderRadius: '10px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '20px',
                color: 'black',
            },
        },
        {
            id: 'node-3',
            type: 'default',
            position: { x: 980, y: 280 },
            data: {},
            style: {
                width: '195px',
                height: '760px',
                backgroundColor: 'transparent', //'#eae4e9'
                border: '3px dashed #ABA1BF',
                borderRadius: '10px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '20px',
                color: 'black',
            },
        },
    ] : [];
    

    // --- Edge ---
    resData.edges.forEach(edge => {
        const priorNodeId = nodeOrderToIdMap[edge.prior];
        const nextNodeId = nodeOrderToIdMap[edge.next];
        if (!priorNodeId || !nextNodeId) {
            console.error(`엣지 ${edge.id}가 존재하지 않는 노드 참조: ${edge.prior}, ${edge.next}`);
            return;
        }

        const priorNode = initElements.find(node => node.id === String(priorNodeId));
        const nextNode  = initElements.find(node => node.id === String(nextNodeId));

        let sourceHandle = 'source-right';  // 기본 값
        let targetHandle = 'target-left';   // 기본 값

        if (priorNode && nextNode) {
            const { x:priorX, y:priorY } = priorNode.position;
            const { x:nextX,  y:nextY  } = nextNode.position;
            if (!isYolo) {
                sourceHandle = 'source-bottom';
                targetHandle = 'target-top';
            } else {
                if (priorX < nextX)      { sourceHandle = 'source-right';  targetHandle = 'target-left';   } 
                else if (priorX > nextX) { sourceHandle = 'source-left';   targetHandle = 'target-right';  } 
                else if (priorY < nextY) { sourceHandle = 'source-bottom'; targetHandle = 'target-top';    } 
                else if (priorY > nextY) { sourceHandle = 'source-top';    targetHandle = 'target-bottom'; }
            }
        }

        const isSpeicalYolo = isYolo && edge.id === 16;
        initElements.push({
            id: String(edge.id),
            source: String(priorNodeId),
            target: String(nextNodeId),
            type: isSpeicalYolo ? "customYOLO" : "custom",
            sourceHandle: isSpeicalYolo ? 'source-right' : sourceHandle,
            targetHandle: isSpeicalYolo ? 'target-top' : targetHandle,
        });
    });

    return isYolo ? [...additionalNodes, ...initElements] : initElements;
}

export default useInitialArch;

