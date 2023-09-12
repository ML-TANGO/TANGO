import React, { useEffect } from "react"
import LoadingOverlay from "react-loading-overlay"
import { ResizableBox } from "react-resizable"
import styled from "styled-components"
import { useWindowSize } from "../Utils/useWindowSize"

import { MdClear } from "react-icons/md"

const HeaderWrapper = styled("div")`
  width: 100%;
  background-color: inherit;
  padding: 0.7rem;
  text-align: left;
`

function CommonPanel({ top, bottom, tail, springProps, title, panelToggle, isSave, loadingText, panelPos, setPanelPos }) {
  const windowSize = useWindowSize()

  useEffect(() => {
    setPanelPos({ top: windowSize.height * 0.4, bottom: windowSize.height * 0.4 })
  }, [windowSize])

  return (
    <React.Fragment>
      <div
        style={{ width: "100vw", height: "100vh", background: "#000", opacity: 0.4, position: "absolute", top: 0, left: 0, zIndex: 998 }}
      ></div>

      <div
        style={{
          // ...springProps,
          background: "black",
          height: "100vh",
          width: "40%",
          position: "absolute",
          top: "0",
          right: "0",
          zIndex: "999",
          overflowY: "auto"
        }}
      >
        <div>
          <HeaderWrapper>
            <div className="d-flex justify-content-between">
              <h4>{title}</h4>
              <MdClear
                className="mr-1 mt-1 hover-pointer"
                style={{ alignSelf: "center", height: "100%" }}
                fontSize={"21"}
                onClick={panelToggle}
              />
            </div>
          </HeaderWrapper>
          <div className="line-separator mx-2 mt-2" />
          <LoadingOverlay active={isSave} spinner text={loadingText}>
            <div style={{ height: `${windowSize.height * 0.85}px` }}>
              <ResizableBox
                width={windowSize.width * 0.4}
                height={windowSize.height * 0.4}
                resizeHandles={["s"]}
                axis="y"
                minConstraints={[200, windowSize.height * 0.2]}
                maxConstraints={[Infinity, windowSize.height * 0.8 * 0.8]}
                handle={
                  <div className="w-100 py-0 px-2 my-3">
                    <div className="line-separator resize-cursor" />
                  </div>
                }
                onResize={(e, data) => {
                  const totalHeight = windowSize.height * 0.8
                  const tabelHeight = totalHeight - data.size.height
                  setPanelPos({ top: data.size.height, bottom: tabelHeight })
                }}
              >
                <div className="px-2 mb-2" style={{ overflowY: "auto", height: panelPos.top }}>
                  {top}
                </div>
              </ResizableBox>
              <div>
                <div className="px-2 mt-4 w-100" style={{ overflowY: "auto", height: `${panelPos.bottom}px` }}>
                  {bottom}
                </div>
              </div>
            </div>
            <div>{tail}</div>
          </LoadingOverlay>
        </div>
      </div>
    </React.Fragment>
  )
}

CommonPanel.propTypes = {}

export default CommonPanel
