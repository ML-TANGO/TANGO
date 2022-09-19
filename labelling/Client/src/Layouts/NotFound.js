import React from "react"
import { Link } from "react-router-dom"

function NotFound() {
  return (
    <div className="theme-light ltr-support without-y-scrollbar">
      <div className="account" style={{ background: "radial-gradient( #364261, #2e2e2e, #000)" }}>
        <div className="m-auto">
          <h1 className="mb-2">404 - Not Found!</h1>
          <Link to="/">Go BluAi Main</Link>
        </div>
      </div>
    </div>
  )
}

export default NotFound
