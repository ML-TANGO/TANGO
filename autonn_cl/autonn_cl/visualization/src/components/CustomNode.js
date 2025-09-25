import React from 'react';
import { Handle, Position } from 'react-flow-renderer';

const CustomNode = ({ data }) => {
  return (
    <div style={{
        background: data.color,
        width: '160px',
        fontSize: "20px",
        fontFamily: "Helvetica",
        boxShadow: "5px 5px 5px 0px rgba(0, 0, 0, .10)",
        borderRadius: "10px",
        border: "none",
        padding: "10px",
        textAlign: "center",
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      }}>
      <Handle type="target" position={Position.Top} id="target-top" style={{ top: -4, background: '#000', zIndex: 10 }} />
      <Handle type="target" position={Position.Right} id="target-right" style={{ right: -4, background: '#000', zIndex: 10 }} />
      <Handle type="target" position={Position.Bottom} id="target-bottom" style={{ bottom: -4, background: '#000', zIndex: 10 }} />
      <Handle type="target" position={Position.Left} id="target-left" style={{ left: -4, background: '#000', zIndex: 10 }} />
      <Handle type="source" position={Position.Top} id="source-top" style={{ top: -4, background: '#000', zIndex: 10 }} />
      <Handle type="source" position={Position.Right} id="source-right" style={{ right: -4, background: '#000', zIndex: 10 }} />
      <Handle type="source" position={Position.Bottom} id="source-bottom" style={{ bottom: -4, background: '#000', zIndex: 10 }} />
      <Handle type="source" position={Position.Left} id="source-left" style={{ left: -4, background: '#000', zIndex: 10 }} />
      <div style={{ width: '100%', textAlign: 'center', margin: 0, padding: 0, boxSizing: 'border-box' }}>{data.label}</div>
    </div>
  );
};

export default CustomNode;
