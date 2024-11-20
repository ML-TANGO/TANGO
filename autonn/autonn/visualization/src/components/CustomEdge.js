import React from 'react';
import { getBezierPath, getMarkerEnd, ArrowHeadType } from 'react-flow-renderer';
import "./styleedge.css";

export default function CustomEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  arrowHeadType = ArrowHeadType.ArrowClosed,
  markerEndId,
}) {
  // 엣지의 비지어 경로를 계산합니다.
  const edgePath = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // 화살표를 설정합니다.
  const markerEnd = getMarkerEnd(arrowHeadType, markerEndId);

  return (
    <>
      <path
        id={id}
        style={style}
        className="react-flow__edge-path"
        d={edgePath}
        markerEnd={markerEnd}
      />
      {data && data.text && (
        <text>
          <textPath
            href={`#${id}`}
            style={{ fontSize: '12px', fill: '#000' }}
            startOffset="50%"
            textAnchor="middle"
          >
            {data.text}
          </textPath>
        </text>
      )}
    </>
  );
}
