import React from "react";
import NodeColorProp from "../../NodeColor";

const LayerToggleYolo = () => {
  const onDragStart = (event, nodeName,  nodeColor,subpm) => {
    event.dataTransfer.setData("application/reactflow", nodeName);
    event.dataTransfer.setData("subparameters", subpm);
    event.dataTransfer.setData('colorNode',nodeColor);
    event.dataTransfer.effectAllowed = "move";
  };
  return (
      <div className="LayerToggle">
          <h2 className="Layer">Layer</h2>
          <aside>
              <details className="categoryConv">
                  <summary className="layerName">ADown</summary>
                  <ul>
                      <li>
                          <div
                              className="dndnode"
                              onDragStart={(event) => onDragStart(event, "ADown", `${NodeColorProp.Yolo_ADown}`, "'dim': 1")}
                              draggable
                          >
                              AConv
                          </div>
                      </li>
                  </ul>
              </details>
              <details className="categoryConv">
                  <summary className="layerName">Conv</summary>
                  <ul>
                      <li>
                          <div
                              className="dndnode"
                              onDragStart={(event) => onDragStart(event, "Conv", `${NodeColorProp.Yolo_Conv}`, "'dim': 1")}
                              draggable
                          >
                              Conv
                          </div>
                      </li>
                  </ul>
              </details>
              <details className="categoryConv">
                  <summary className="layerName">RepNCSPELAN4</summary>
                  <ul>
                      <li>
                          <div
                              className="dndnode"
                              onDragStart={(event) => onDragStart(event, "RepNCSPELAN4", `${NodeColorProp.Yolo_RepNCSPELAN4}`, "'dim': 1")}
                              draggable
                          >
                              RepNCSPELAN4
                          </div>
                      </li>
                  </ul>
              </details>
              <details className="categoryConv">
                  <summary className="layerName">SPPELAN</summary>
                  <ul>
                      <li>
                          <div
                              className="dndnode"
                              onDragStart={(event) => onDragStart(event, "SPPELAN", `${NodeColorProp.Yolo_SPPELAN}`, "'dim': 1")}
                              draggable
                          >
                              SPPELAN
                          </div>
                      </li>
                  </ul>
              </details>
              <details className="categoryConv">
                  <summary className="layerName">Upsample</summary>
                  <ul>
                      <li>
                          <div
                              className="dndnode"
                              onDragStart={(event) => onDragStart(event, "Upsample", `${NodeColorProp.Yolo_Upsample}`, "'dim': 1")}
                              draggable
                          >
                              Upsample
                          </div>
                      </li>
                  </ul>
              </details>
              <details className="categoryConv">
                  <summary className="layerName">Concat</summary>
                  <ul>
                      <li>
                          <div
                              className="dndnode"
                              onDragStart={(event) => onDragStart(event, "Concat", `${NodeColorProp.Yolo_Concat}`, "'dim': 1")}
                              draggable
                          >
                              Concat
                          </div>
                      </li>
                  </ul>
              </details>
              <details className="categoryConv">
                  <summary className="layerName">Detect</summary>
                  <ul>
                      <li>
                          <div
                              className="dndnode"
                              onDragStart={(event) => onDragStart(event, "Detect", `${NodeColorProp.Yolo_Detect}`, "'dim': 1")}
                              draggable
                          >
                              Concat
                          </div>
                      </li>
                  </ul>
              </details>
          </aside>
      </div>
  );
};

export default LayerToggleYolo;
