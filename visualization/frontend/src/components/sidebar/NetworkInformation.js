import React from "react";
import YOLOv5 from "../../img/YOLOv5.png"
const NetworkInformation = () => {
  const onDragStart = (event, nodeName,  nodeColor,subpm) => {
    event.dataTransfer.setData("application/reactflow", nodeName);
    event.dataTransfer.setData("subparameters", subpm);
    event.dataTransfer.setData('colorNode',nodeColor);
    event.dataTransfer.effectAllowed = "move";
  };
  return (
      <div className="NetworkInformation">
          <h2 className="InfoText">Network Information</h2>

          <div className="NetworkInformationDiv">
              <aside>
                  <div className="YoloText">
                      YOLOv5
                  </div>
                  <img className="yolo_image" src={YOLOv5} alt="yolov5"/>
              </aside>
          </div>

      </div>
  );
};

export default NetworkInformation;