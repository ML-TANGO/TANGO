import React from 'react';
import { getMarkerEnd, ArrowHeadType } from 'react-flow-renderer';
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
  // 엣지 경로를 정의합니다. 이 예제에서는 Y 좌표는 sourceY에서 targetY로 바로 가고, X 좌표만 변경합니다.
  const edgePath = `
    M ${sourceX},${sourceY}
    L ${targetX},${sourceY}
    L ${targetX},${targetY}
  `;

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
