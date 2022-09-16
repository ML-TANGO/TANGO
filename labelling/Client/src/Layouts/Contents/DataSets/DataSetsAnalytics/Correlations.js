import React, { useRef, useEffect } from "react"
import { Row, Col } from "reactstrap"
import * as d3 from "d3"
import d3Tip from "d3-tip"

const Correlations = ({ data }) => {
  const svgRef = useRef()
  useEffect(() => {
    let svg
    if (data?.GRAPH_TYPE === "CORRELATION GRAPH") {
      let margin = { top: 30, right: 70, bottom: 150, left: 180 }
      let width = data.LEGEND_X.length * 100 >= 600 ? 600 : data.LEGEND_X.length * 100
      let height = data.LEGEND_Y.length * 100 >= 600 ? 600 : data.LEGEND_Y.length * 100
      let result = []
      data.LEGEND_X.forEach((x, i) => {
        data.LEGEND_Y.forEach((y, j) => {
          result.push({ x: x, y: y, value: data.GRAPH_DATA[0].GRAPH_POSITION[i][j] })
        })
      })
      svg = d3.select(svgRef.current)

      svg
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")

      // y축 그리기
      let y = d3.scaleBand().range([0, height]).domain(data.LEGEND_Y).padding(0.01)
      svg.append("g").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y)).attr("font-size", "12px")

      // x축 그리기
      let x = d3.scaleBand().range([0, width]).domain(data.LEGEND_X).padding(0.01)
      svg
        .append("g")
        .attr("transform", `translate(${margin.left},${height})`)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .style("text-anchor", "end")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-65)")
        .attr("font-size", "12px")

      // var myColor = d3.scaleLinear().domain([0, 1]).range(["rgb(255,255,255)", "rgb(0, 151, 230)"])
      const myColor = d3.scaleLinear().domain([-1, 0, 1]).range(["rgb(255,0,0)", "rgb(255,255,255)", "rgb(0, 151, 230)"])

      const tip = d3Tip()
        .attr("class", "d3-tip-alpha")
        .style("background", "black")
        .style("border", "1px solid #ddd")
        .style("padding", "5px")
        .offset(() => [-10, 0])
        .html((d, v) => `X : ${v.x} Y : ${v.y} </br> ${v.value ? Number(v.value).toFixed(4) : "NaN"}`)

      // draw rect
      svg
        .selectAll()
        .data(result)
        .enter()
        .append("rect")
        .classed("rect", true)
        .attr("transform", "translate(" + margin.left + ",0)")
        .attr("x", d => x(d.x))
        .attr("y", d => y(d.y))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .style("fill", d => myColor(d.value))
        .call(tip)
        .on("mouseover", tip.show)
        .on("mouseleave", tip.hide)

      ///////////////////////////////////////////////////////////////////////////
      //////////////// Create the gradient for the legend ///////////////////////
      ///////////////////////////////////////////////////////////////////////////

      //Extra scale since the color scale is interpolated
      const countScale = d3
        .scaleLinear()
        .domain([-1, 0, 1])
        .range([0, width / 2, width])

      //Calculate the variables for the temp gradient
      const numStops = 20
      let countRange = countScale.domain()
      countRange[3] = countRange[2] - countRange[0]
      let countPoint = []
      for (var i = 0; i < numStops; i++) {
        countPoint.push((i * countRange[3]) / (numStops - 1) + countRange[0])
      } //for i

      //Create the gradient
      svg
        .append("defs")
        .append("linearGradient")
        .attr("id", "legend-traffic")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%")
        .selectAll("stop")
        .data(d3.range(numStops))
        .enter()
        .append("stop")
        .attr("offset", function (d, i) {
          return countScale(countPoint[i]) / width
        })
        .attr("stop-color", function (d, i) {
          return myColor(countPoint[i])
        })

      ///////////////////////////////////////////////////////////////////////////
      ////////////////////////// Draw the legend ////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////

      const legendWidth = width * 0.95
      //Color Legend container
      const legendsvg = svg
        .append("g")
        .attr("class", "legendWrapper")
        .attr("transform", `translate(${width + margin.left + 15},${height / 2}) rotate(-90)`)
      // .attr("transform", "rotate(-90)")

      //Draw the Rectangle
      legendsvg
        .append("rect")
        .attr("class", "legendRect")
        .attr("x", -legendWidth / 2)
        .attr("y", 0)
        //.attr("rx", hexRadius*1.25/2)
        .attr("width", legendWidth)
        .attr("height", 20)
        .style("fill", "url(#legend-traffic)")

      //Set scale for x-axis
      const xScale = d3
        .scaleLinear()
        .domain([-1, 0, 1])
        .range([-legendWidth / 2, 0, legendWidth / 2])

      //Define x-axis
      const xAxis = d3.axisBottom().ticks(3).scale(xScale)

      //Set up X axis
      legendsvg
        .append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,20)`)
        .call(xAxis)
        .selectAll("text")
        .style("text-anchor", "start")
        .attr("y", 0)
        .attr("x", 9)
        .attr("dy", "0.35em")
        .attr("transform", "rotate(90)")
    }
    return () => {
      if (data?.GRAPH_TYPE === "CORRELATION GRAPH") {
        svg = d3.select(svgRef.current)
        svg.selectAll("svg > *").remove()
      }
    }
  }, [data])
  return (
    <Row>
      <Col>
        <div className="line-separator mr-2 ml-2" />
        <h5>{data.GRAPH_DATA[0].GRAPH_NAME}</h5>
        <div className="line-separator mr-2 ml-2 mb-3" />
        <svg ref={svgRef} />
      </Col>
    </Row>
  )
}

export default Correlations
