import React, { useState, useRef, useEffect } from "react";
import "../../styles.css";
import CustomEdge from "../CustomEdge";
import CustomEdgeYOLO from "../CustomEdgeYOLO";
import EditModal from "../layer/PopupModal"; /* Conv2d */
import Conv from "../layer/Conv"; /* Conv2d + BatchNorm2d + SiLU */
import MaxPoolModal from "../layer/MaxPool";
import AvgPool2d from "../layer/AvgPool2d";
import AdaptiveAvgPool2d from "../layer/AdaptiveAvgPool2d";
import MP from "../layer/MP";
import SP from "../layer/SP";
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
import Concat from "../layer/Concat";
import Shortcut from "../layer/Shortcut";
import DownC from "../layer/DownC";
import SPPCSPC from "../layer/SPPCSPC";
import ReOrg from "../layer/ReOrg";
import IDetect from "../layer/IDetect";
import axios from "axios";
import ReactFlow, {
  addEdge,
  MiniMap,
  ReactFlowProvider,
  removeElements,
  Controls,
  ControlButton,
} from "react-flow-renderer";
import GenerateButton from "../GenerateButton";
import Tab from "../sidebar/Tab";
import LayerToggle from "../sidebar/LayerToggle";
import NetworkInformation from "../sidebar/NetworkInformation";
import InitialArch from "../../InitialArch"; // 수정된 InitialArch.js
import arange_icon from "../../img/swap.png";

// 추가된 이미지 import 구문
import SPPELAN from "../../img/SPPELAN.png";
import RepNCSPELAN4 from "../../img/RepNCSPELAN4.png";
import ADown from "../../img/ADown.png";
import yolo_Conv from "../../img/Conv.png";

import BasicBlockimg from "../../img/basicblock.png";
import BottleNeckimg from "../../img/bottleneck.png";
import CustomNode from "../CustomNode";

let id = 1;
const getId = () => `${id}`; // ID 자동 증가
let nowc = 0;
const edgeTypes = {
  custom: CustomEdge,
  customYOLO: CustomEdgeYOLO,
};
const nodeTypes = {
  custom: CustomNode,
};

let nowp = "";
var checkFirst = 0;
let initRunningStateTime = 100;
var running_id = 0;
var sortCount = 1;
var sortHeight = 0;
let sortList = [];
let clickedNodeList = [];
let clickedNodeIdList = [];

function LayerList() {
  const [isYolo, setIsYolo] = useState(false); // YOLO 모드를 위한 상태 추가
  const [taskType, setTaskType] = useState(""); // task_type을 처리하기 위한 상태
  const [isInitialLoading, setIsInitialLoading] = useState(true); // 로딩 상태 추가
  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [state, setState] = useState("");
  const [idState, setIdState] = useState("");
  const [paramState, setParam] = useState();
  const [group, setGroup] = useState(false);
  const [level, setLevel] = useState(1);
  const [ungroup, setUngroup] = useState(false);
  const [isSort, setIsSort] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [hoverImage, setHoverImage] = useState(null);

  useEffect(() => {
    // task_type에 따라 isYolo 설정 (서버에서 task_type을 가져와 설정)
    const fetchTaskType = async () => {
      try {
        const response = await axios.get("/api/info/"); // /api/info/ 에서 task 값 가져옴
        //const taskType = response.data[response.data.length - 1].task;
        const taskType = response.data[response.data.length - 1].model_type;
        setTaskType(taskType);
        //setIsYolo(taskType.toLowerCase() === "detection");
        setIsYolo(taskType.toLowerCase() === "yolov9");
      } catch (error) {
        console.error("Error fetching task_type:", error);
      }
    };

    fetchTaskType();
  }, []);

  // 노드 및 엣지 데이터를 가져오는 부분 (isYolo, taskType 상태에 따른 처리)
  const [elements, setElements, isLoading] = InitialArch(
    level,
    group,
    setGroup,
    ungroup,
    setUngroup,
    isSort,
    setIsSort,
    isYolo
  );

  useEffect(() => {
    const get_params = async () => {
      try {
        await axios
          .get("/api/node/".concat(String(idState)).concat("/"))
          .then((response) => {
            setParam(response.data.parameters);
          });
      } catch (error) {
        console.error(error);
      }
    };
    get_params();
  }, [idState]);

  const onConnect = async (params) => {
    if (params.source === params.target) {
      return;
    }

    const get_edge = async () => {
      try {
        return await axios.get("/api/edge/");
      } catch (error) {
        console.error(error);
      }
    };

    const cedge = await get_edge();
    var maxId = 0;
    for (var i = 0; i < cedge.data.length; i++) {
      if (maxId < cedge.data[i].id) {
        maxId = cedge.data[i].id;
      }
    }

    const newEdge = {
      id: maxId + 1,
      source: params.source,
      target: params.target,
      type: isYolo ? "customYOLO" : "custom", // isYolo에 따라 엣지 타입 설정
    };

    setElements((els) => addEdge(newEdge, els));

    axios
      .post("/api/edge/", {
        id: maxId + 1,
        prior: params.source,
        next: params.target,
      })
      .then(function (response) {
        console.log(response);
      })
      .catch((err) => console.log(err));
  };

  const onElementsRemove = (remove) => {
    setElements((els) => removeElements(remove, els));
  };

  const openModal = async () => {
    setModalOpen(true);
    console.log("open modal");
  };

  const closeModal = () => {
    setModalOpen(false);
  };

  const saveModal = () => {
    setModalOpen(false);
  };

  const onNodeClick = async (event, node) => {
    await setState(node.data.label);
    await setIdState(node.id);

    switch (node.data.label) {
      case "SPPELAN":
        setHoverImage(SPPELAN);
        break;
      case "RepNCSPELAN4":
        setHoverImage(RepNCSPELAN4);
        break;
      case "ADown":
        setHoverImage(ADown);
        break;
      case "Conv":
        setHoverImage(yolo_Conv);
        break;
      default:
        setHoverImage(null);
    }
  };

  const C = () => {
    const components = {
      Conv2d: EditModal,
      Conv: Conv,
      MaxPool2d: MaxPoolModal,
      AvgPool2d: AvgPool2d,
      AdaptiveAvgPool2d: AdaptiveAvgPool2d,
      MP: MP,
      SP: SP,
      Softmax: Softmax,
      ConstantPad2d: ConstantPad2d,
      BatchNorm2d: BatchNorm2d,
      MSELoss: MSELoss,
      Tanh: Tanh,
      Sigmoid: Sigmoid,
      CrossEntropyLoss: CrossEntropyLoss,
      Linear: Linear,
      Dropout: Dropout,
      ZeroPad2d: ZeroPad2d,
      BCELoss: BCELoss,
      LeakyReLU: LeakyReLU,
      ReLU: ReLU,
      ReLU6: ReLU6,
      Flatten: Flatten,
      ReOrg: ReOrg,
      BasicBlock: BasicBlock,
      Bottleneck: Bottleneck,
      Concat: Concat,
      Shortcut: Shortcut,
      DownC: DownC,
      SPPCSPC: SPPCSPC,
      IDetect: IDetect,
      Upsample: Upsample,
    };

    const Component = components[state] || null;
    return Component ? (
      <Component
        params={paramState}
        layer={idState}
        open={modalOpen}
        save={saveModal}
        close={closeModal}
        header={state}
        setState={setIdState}
      />
    ) : null;
  };

  const [tabToggle, setTabtoggle] = useState(1);
  const tabOnClick = (path) => {
    if (path === "info icon") {
      setTabtoggle(2);
    } else {
      setTabtoggle(1);
    }
  };

  if (isLoading) {
    return <div>로딩중...</div>;
  }

  return (
    <div className="FullPage">
      <div className="Sidebar">
        <Tab tabOnClick={tabOnClick} />
        {tabToggle === 1 ? (
          <LayerToggle isYolo={isYolo} setIsYolo={setIsYolo} />
        ) : (
          <NetworkInformation />
        )}
        <div className="LayerInfo">
          <h3>Layer Information</h3>
          <C />
        </div>
      </div>

      <div className="dndflow">
        <ReactFlowProvider>
          <div className="reactflow-wrapper" ref={reactFlowWrapper}>
            <ReactFlow
              onConnect={onConnect}
              elements={elements}
              onLoad={setReactFlowInstance}
              onElementClick={onNodeClick}
              onElementsRemove={onElementsRemove}
              edgeTypes={edgeTypes}
              nodeTypes={nodeTypes}
              key="edges"
            >
              <Controls showZoom showInteractive showFitView>
                <ControlButton title="Sort">
                  <img src={arange_icon} alt="Sort" />
                </ControlButton>
              </Controls>
              <div className="reactBtn" style={{ position: "absolute", zIndex: 100 }}>
                <GenerateButton elements={elements} />
              </div>
              {hoverImage && (
                <div
                  className="hoverImage"
                  style={{
                    position: "fixed",
                    top: 0,
                    left: 0,
                    width: "100vw",
                    height: "100vh",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    backgroundColor: "rgba(0, 0, 0, 0.5)",
                    zIndex: 1000,
                  }}
                  onClick={() => setHoverImage(null)}
                >
                  <div
                    className="hoverImage"
                    style={{
                      position: "relative",
                      zIndex: 1001,
                      pointerEvents: "auto",
                    }}
                  >
                    <img
                      src={hoverImage}
                      alt="Layer Detail"
                      style={{
                        maxWidth: "500px",
                        maxHeight: "500px",
                        cursor: "pointer",
                      }}
                      onClick={() => setHoverImage(null)}
                    />
                  </div>
                </div>
              )}
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

