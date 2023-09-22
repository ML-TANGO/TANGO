import React from "react";
const AbstractNetwork = () => {
  const onDragStart = (event, nodeName,  nodeColor,subpm) => {
    event.dataTransfer.setData("application/reactflow", nodeName);
    event.dataTransfer.setData("subparameters", subpm);
    event.dataTransfer.setData('colorNode',nodeColor);
    event.dataTransfer.effectAllowed = "move";
  };
  return (
      <div className="AbstractNetwork">
          <h2 className="AbstractText">Abstract Network</h2>

          <aside>
              <div className="AutoGroup">
                  <div className="GroupText"> Auto Group </div>
                  <button type="button" className="AbstractBtn"> Level 1 </button>
                  <button type="button" className="AbstractBtn"> Level 2 </button>
                  <button type="button" className="AbstractBtn"> Level 3 </button>
              </div>
              <div className="CustomGroup">
                  <div className="GroupText"> Custom Group </div>
                  <button type="button" className="AbstractBtn"> Group </button>
                  <button type="button" className="AbstractBtn"> Ungroup </button>
              </div>
              <div className="GroupInformation">
                  <div className="GroupText"> Group Information </div>
              </div>


          </aside>
      </div>
  );
};

export default AbstractNetwork;
