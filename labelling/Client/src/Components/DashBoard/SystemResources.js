import React from "react"
import { Pie, PieChart, ResponsiveContainer } from "recharts"
import { Card, CardBody } from "reactstrap"

const SystemResources = props => {
  return (
    <Card className="system-resource-card">
      <CardBody className="card-bg-dark">
        <div className="dashboard__weekly-stat h-100">
          <div className="dashboard__weekly-stat-chart h-100">
            <div className="dashboard__weekly-stat-chart-item">
              <div className="dashboard__weekly-stat-info stop-dragging">
                <p>CPU</p>
              </div>
              <div className="dashboard__weekly-stat-chart-pie ">
                <ResponsiveContainer height={250}>
                  <PieChart>
                    <Pie data={props.value.CPU} dataKey="value" innerRadius={50} outerRadius={70} startAngle={180} endAngle={0} />
                  </PieChart>
                </ResponsiveContainer>
                <p className="dashboard__weekly-stat-label stop-dragging">{props.value.CPU_USED}%</p>
              </div>
            </div>
            <div className="dashboard__weekly-stat-chart-item">
              <div className="dashboard__weekly-stat-info stop-dragging">
                <p>MEMORY</p>
              </div>
              <div className="dashboard__weekly-stat-chart-pie">
                <ResponsiveContainer height={250}>
                  <PieChart>
                    <Pie data={props.value.MEM} dataKey="value" innerRadius={50} outerRadius={70} startAngle={180} endAngle={0} />
                  </PieChart>
                </ResponsiveContainer>
                <p className="dashboard__weekly-stat-label stop-dragging">{props.value.MEM_USED}%</p>
              </div>
            </div>
            <div className="dashboard__weekly-stat-chart-item">
              <div className="dashboard__weekly-stat-info stop-dragging">
                <p>DISK</p>
              </div>
              <div className="dashboard__weekly-stat-chart-pie">
                <ResponsiveContainer height={250}>
                  <PieChart>
                    <Pie data={props.value.DISK} dataKey="value" innerRadius={50} outerRadius={70} startAngle={180} endAngle={0} />
                  </PieChart>
                </ResponsiveContainer>
                <p className="dashboard__weekly-stat-label stop-dragging">{props.value.DISK_USED}%</p>
              </div>
            </div>
          </div>
        </div>
      </CardBody>
    </Card>
  )
}
SystemResources.propTypes = {}

export default SystemResources
