import React, { PureComponent } from "react"
import { withRouter } from "react-router-dom"
import PropTypes from "prop-types"

class MainWrapper extends PureComponent {
  static propTypes = {
    children: PropTypes.any.isRequired
  }

  render() {
    const { children, className } = this.props

    return (
      <div id="wrapper" className={`theme-light ltr-support ${className}`} style={{ minHeight: "100vh" }} dir="ltr">
        <div className={"wrapper"} style={{ minHeight: "100vh" }}>
          {children}
        </div>
      </div>
    )
  }
}

export default withRouter(MainWrapper)
