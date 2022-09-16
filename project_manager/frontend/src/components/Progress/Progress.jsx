import React from "react";
import "./Progress.css";
function Progress({ percent }) {
  return (
    <div>
      <div className="progressBackground">
        <div className="progressValue" style={{ width: `calc(100% * ${percent})` }}></div>
      </div>
    </div>
  );
}

export default Progress;
