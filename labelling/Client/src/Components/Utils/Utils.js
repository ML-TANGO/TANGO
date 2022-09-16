export function getDataType(type) {
  switch (type) {
    case "I":
      return "Image"
    case "V":
      return "Video"
    case "R":
      return "Real-Time"
  }
}

export function getObjectType(type) {
  switch (type) {
    case "C":
      return "Classification"
    case "D":
      return "Detection"
    case "S":
      return "Segmentation"
    case "R":
      return "Regression"
  }
}

export function convertString(str) {
  const temp = str.split("_")
  const type = temp[0]
  const s = temp[1]
  switch (s) {
    case "I":
      return "Image"
    case "V":
      return "Video"
    case "T":
      return "Text"
    case "D":
      return "Detection"
    case "C":
      return "Classification"
    case "S":
      return "Segmentation"
    case "R":
      return type === "D" ? "Real-Time" : "Regression"
  }
}

export function getColor(type) {
  switch (type) {
    case "C":
      return "rgb(65, 125, 255)"
    case "D":
      return "orange"
    case "S":
      return "rgb(10, 208, 0)"
    case "R":
      return "#838383"
  }
}

export function getSchedule(data) {
  switch (data.SCH_WEEKOPTION) {
    case "daily":
      return `매일 ${data.SCH_TIME}`
    case "weekday":
      return `평일 ${data.SCH_TIME}`
    case "weekend":
      return `주말 ${data.SCH_TIME}`
    case "days":
      const arr = JSON.parse(data.SCH_OPTION2).map(ele => getDay(ele))
      return `${arr.join(",")} ${data.SCH_TIME}`
    case "monthly":
      return `매월 ${data.SCH_OPTION2}일 ${data.SCH_TIME}`
  }
}

export function getDay(day) {
  switch (day) {
    case "1":
      return "월"
    case "2":
      return "화"
    case "3":
      return "수"
    case "4":
      return "목"
    case "5":
      return "금"
    case "6":
      return "토"
    case "7":
      return "일"
  }
}

export function getAiStatus(status) {
  let result = ""
  switch (status) {
    case "NONE":
      result = "IDLE"
      break
    case "READY":
      result = "Creating"
      break
    case "FAIL":
      result = "Fail"
      break
    case "LEARN":
      result = "Training"
      break
    case "DONE":
      result = "Done"
      break
  }
  return result
}

export function getStatus(status) {
  switch (status) {
    case "DELETE":
      return "Delete"
    case "CREATE":
      return "Create"
    case "DEL_FAIL":
      return "Delete Fail"
    case "CRN_FAIL":
      return "Create Fail"
    case "AUTO_FAIL":
      return "Auto Labeling Fail"
    case "AUTO":
      return "Auto Labeling"
    case "ACT_FAIL":
      return "Source Excute Fail"
    case "DONE":
      return "Done"
    default:
      return ""
  }
}

export function getDistance(sx, sy, ex, ey, ppm) {
  const dtx = ex - sx
  const dty = ey - sy
  if (ppm !== undefined) return Math.sqrt(Math.pow(dtx, 2) + Math.pow(dty, 2)) * ppm
  else return Math.sqrt(Math.pow(dtx, 2) + Math.pow(dty, 2))
}

export function getLineDistance(points, ppm) {
  if (points.length < 2) return null
  let tot_distance = 0
  for (let i = 1; i < points.length; i++) {
    let distance = getDistance(points[i - 1].X, points[i - 1].Y, points[i].X, points[i].Y, ppm)
    tot_distance += distance
  }
  return Number(tot_distance).toFixed(2)
}

export function _getNeedCount(btnSts) {
  switch (btnSts) {
    case "isRect":
    case "isMagic":
    case "isTracker":
    case "isCross":
    case "isCircle":
      return 2
    case "isAngle":
      return 3
    case "isPolygon":
    case "isBrush":
    case "isDistance":
    case "isArea":
      return -1
  }
}

/*
Get cross line
return line points [sx, sy, ex, ey]
point : [{X:x1, Y:y1},{X:x2, Y:y2}]
*/
export function _getOrthoLine(point) {
  if (point === undefined) return null
  if (point.length < 2) return null
  // 시작 포인트 값
  let x0 = point[0].X
  let y0 = point[0].Y
  // 종료 포인트 값
  let x1 = point[1].X
  let y1 = point[1].Y

  // 기준 라인의 기울기
  let a = x1 - x0 === 0 ? 0 : (y1 - y0) / (x1 - x0)
  // 거리 결정 상수
  let d = 40

  // 거리에 대한 제한 갈이가 있는 경우 (ex : 2mm)
  // if (width != undefined && width != "") d = (dpi * width) / 2 / 25.4

  // 각도 결정 상수
  let base = 90 * (Math.PI / 180)

  // 기준 좌표의 끼인각 크기 구하기
  let theta = Math.atan((y1 - y0) / (x1 - x0))

  // 기울기가 0인 경우 처리 및 d 만큼 떨어진 곳의 좌표 구하기
  // x1 = x0 + d * cos(angle)
  // y1 = y0 + d * sin(angle)
  let x2 = a === 0 ? (x0 === x1 ? x1 + d : x1) : x1 + d * Math.cos(theta - base)
  let y2 = a === 0 ? (y0 === y1 ? y1 + d : y1) : y1 + d * Math.sin(theta - base)
  let x3 = a === 0 ? (x0 === x1 ? x1 - d : x1) : x1 + d * Math.cos(theta + base)
  let y3 = a === 0 ? (y0 === y1 ? y1 - d : y1) : y1 + d * Math.sin(theta + base)

  // 보정 계수를 위한 교점 구하기
  let interPoint = [(x2 + x3) / 2, (y2 + y3) / 2]
  // 보정계수 구하기
  let coefficientX = x1 - interPoint[0]
  let coefficientY = y1 - interPoint[1]
  return [x2 + coefficientX, y2 + coefficientY, x3 + coefficientX, y3 + coefficientY]
}

/*
Get radius from 2 points
return radius
points = [{X:x1, Y:y1},{X:x2, Y:y2}]
*/
export function _getRadius(points) {
  return Math.sqrt(Math.pow(points[0].X - points[1].X, 2) + Math.pow(points[0].Y - points[1].Y, 2))
}

/*
Get angle from 3 points
return angle
point : [{X:x1, Y:y1},{X:x2, Y:y2},{X:x1, Y:y1},{X:x2, Y:y2}]
pointSort : true / flase
*/
export function _getAngle(points, pointSort) {
  if (points === undefined) return null
  if (points.length < 3) return null
  let cx, cy, x1, y1, x2, y2
  cx = points[0].X
  cy = points[0].Y
  x1 = points[1].X
  y1 = points[1].Y
  x2 = points[2].X
  y2 = points[2].Y
  if (pointSort) {
    x1 = points[0].X
    y1 = points[0].Y
    cx = points[1].X
    cy = points[1].Y
    x2 = points[2].X
    y2 = points[2].Y
  }

  const a = Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2))
  const b = Math.sqrt(Math.pow(x1 - cx, 2) + Math.pow(y1 - cy, 2))
  const c = Math.sqrt(Math.pow(cx - x2, 2) + Math.pow(cy - y2, 2))

  const temp = (Math.pow(b, 2) + Math.pow(c, 2) - Math.pow(a, 2)) / (2 * b * c)

  let Angle = Math.acos(temp.toFixed(10))
  Angle = Angle * (180 / Math.PI)
  return Number(Angle.toFixed(2))
}

/*
Get rotation
return rotation
angle :
  undefined =  point : [{X:x1, Y:y1},{X:x2, Y:y2},{X:x3, Y:y3}]
angle :
  defined =  point : [{X:x1, Y:y1},{X:x2, Y:y2}]
*/
export function _getRotation(points, sort, angle) {
  let cx, cy, x1, y1, x2, y2
  cx = points[0].X
  cy = points[0].Y
  x1 = points[1].X
  y1 = points[1].Y
  x2 = points[2].X
  y2 = points[2].Y

  if (sort) {
    x1 = points[0].X
    y1 = points[0].Y
    cx = points[1].X
    cy = points[1].Y
    x2 = points[2].X
    y2 = points[2].Y
  }

  const dx = getDistance(x1, cy, cx, cy)
  const dy = getDistance(x1, y1, x1, cy)
  let rotation = Math.atan(dy / dx)
  if (angle === undefined) {
    const angleVal = _getAngle(points, sort)

    let rt = 0
    if (x1 > cx && x2 > cx) {
      if (y1 < y2) rt = 0
      else rt = angleVal
    } else if (x1 < cx && x2 < cx) {
      if (y1 < y2) rt = angleVal
      else rt = 0
    } else if (y1 < cy && y2 < cy) {
      if (x1 < x2) rt = 0
      else rt = angleVal
    } else if (y1 > cy && y2 > cy) {
      if (x1 > x2) rt = 0
      else rt = angleVal
    } else {
      let a = (y1 - y2) / (x1 - x2)
      let b = y1 - a * x1
      let x = (cy - b) / a
      if (x > cx) {
        if (cy < y1) rt = angleVal
        else rt = 0
      } else {
        if (cy < y1) rt = 0
        else rt = angleVal
      }
    }

    if (x1 >= cx && y1 <= cy) {
      // 1사분면
      rotation = 360 - (rotation * 180) / Math.PI - rt
    } else if (x1 >= cx && y1 >= cy) {
      // 4사분면
      rotation = (rotation * 180) / Math.PI - rt
    } else if (x1 <= cx && y1 >= cy) {
      // 3사분면
      rotation = 180 - (rotation * 180) / Math.PI - rt
    } else {
      rotation = (rotation * 180) / Math.PI + 180 - rt
    }
  } else {
    if (x1 >= cx && y1 <= cy) {
      // 1사분면
      rotation = 360 - (rotation * 180) / Math.PI - angle / 2
    } else if (x1 >= cx && y1 >= cy) {
      // 4사분면
      rotation = (rotation * 180) / Math.PI - angle / 2
    } else if (x1 <= cx && y1 >= cy) {
      // 3사분면
      rotation = 180 - (rotation * 180) / Math.PI - angle / 2
    } else {
      rotation = (rotation * 180) / Math.PI + 180 - angle / 2
    }
  }

  return rotation
}

export function _getArcDistance(points) {
  let cx, cy, x1, y1, x2, y2
  x1 = points[0].X
  y1 = points[0].Y
  cx = points[1].X
  cy = points[1].Y
  x2 = points[2].X
  y2 = points[2].Y

  const x1Distance = getDistance(x1, y1, cx, cy)
  const x2Distance = getDistance(x2, y2, cx, cy)
  let distance
  x1Distance <= x2Distance ? (distance = x1Distance) : (distance = x2Distance)

  return distance / 5
}

/*
Get wide in selected area
return wide
points : [ {X:x1, Y:y1}, {X:x2, Y:y2}, {X:x3, Y:y3}, ... , {X:xn, Y:yn}]
dpi :
  undefined = not supported
  value = return real distance in millimeter
*/
export function _getArea(points, ppm) {
  // const oneDpi = 0.393701
  // const px4mm = (1 / (dpi * oneDpi)) * 10

  let tot_Area = 0
  let j = points.length - 1

  for (let i = 0; i < points.length; i++) {
    tot_Area = tot_Area + (points[j].X + points[i].X) * ppm * ((points[j].Y - points[i].Y) * ppm)
    j = i
  }
  return Math.abs(tot_Area / 2).toFixed(2)
}

// 데이터 샘플링
export function _sampling(data) {
  let d = 5 // 이 거리에 따라 포인트수가 조정됨
  let newData = []
  let prev = { X: 0, Y: 0 }
  data.map(el => {
    if (getDistance(prev.X, prev.Y, el.X, el.Y) >= d) {
      newData.push(el)
      prev = el
    }
  })
  return newData
}

export function bytesToSize(bytes) {
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"]
  if (!bytes) {
    return "-"
  }
  const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)))
  if (i === 0) {
    return bytes + " " + sizes[i]
  }

  return (bytes / Math.pow(1024, i)).toFixed(1) + " " + sizes[i]
}

export function isNumber(str) {
  return !isNaN(Number(str))
}
