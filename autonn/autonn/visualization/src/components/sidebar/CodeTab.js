import React from "react";
const CodeTab = () => {
  const onDragStart = (event, nodeName,  nodeColor, subpm) => {
    event.dataTransfer.setData("application/reactflow", nodeName);
    event.dataTransfer.setData("subparameters", subpm);
    event.dataTransfer.setData('colorNode',nodeColor);
    event.dataTransfer.effectAllowed = "move";
  };
  return (
      <div className="CodeTab">
          <h2 className="CodeText"> Code </h2>

          <aside>
              <div className="Framework">
                  <div className="FrameworkText"> Framework </div>
                  <button type="button" className="FrameworkBtn"> Pytorch </button>
                  <button type="button" className="FrameworkBtn"> Tensorflow </button>
              </div>
              <div className="Code">
                  <div className="CodeGroupText"> Code </div>
                  <div className="CodeDiv">

                  </div>
              </div>
          </aside>
      </div>
  );
};

export default CodeTab;
