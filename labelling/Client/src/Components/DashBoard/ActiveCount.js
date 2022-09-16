import React, { useState } from "react"
import { Card, CardBody } from "reactstrap"
import { FaRunning } from "react-icons/fa"

const ActiveModels = props => {
  const [egg, setEgg] = useState(false)

  return (
    <Card>
      <CardBody className="active-count-card-front card-bg-light">
        <div className="card__title">
          <div className="card__title-center stop-dragging">{props.title}</div>
        </div>

        <div className="dashboard__health-chart-info">
          <p className="dashboard__health-chart-number stop-dragging">
            <FaRunning
              className={`dashboard__health-chart-icon ${egg ? "blinking-egg" : "blinking"}`}
              onDoubleClick={() => {
                setEgg(egg => !egg)
              }}
            />
            {props.model[0]?.value !== undefined && props.model[0]?.value}
          </p>
          <p className="dashboard__health-chart-units stop-dragging">
            {props.model[1]?.value !== undefined && `${props.model[1]?.label} : ${props.model[1]?.value}`}
          </p>
          <p className="dashboard__health-chart-units stop-dragging">
            {props.model[2]?.value !== undefined && `${props.model[2]?.label} : ${props.model[2]?.value}`}
          </p>
        </div>
      </CardBody>
    </Card>
  )
}

ActiveModels.propTypes = {}

export default ActiveModels
