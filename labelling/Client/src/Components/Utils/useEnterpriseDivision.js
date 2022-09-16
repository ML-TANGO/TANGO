import React from "react"
import { FiFileText, FiImage, FiVideo, FiTv } from "react-icons/fi"

const c = {
  dataSet: {
    dataSetUpload: {
      CE: { I: { COUNT: 500, SIZE: Infinity }, V: { COUNT: 1, SIZE: 524288000 }, T: { COUNT: 500, SIZE: 524288000 } },
      EE: { I: { COUNT: Infinity, SIZE: Infinity }, V: { COUNT: Infinity, SIZE: Infinity }, T: { COUNT: Infinity, SIZE: Infinity } }
    },
    createDataSet: { CE: 3, EE: Infinity },
    dataSetDuplication: { CE: false, EE: true },
    labelTracker: { CE: false, EE: true },
    autoUserModel: { CE: false, EE: true }
  },
  aiModel: {
    createModel: { CE: 3, EE: Infinity },
    autoMl: {
      CE: { C: false, D: true, S: true },
      EE: { C: false, D: false, S: false }
    },
    transfer: { CE: true, EE: false },
    multiGpu: { CE: false, EE: true },
    startModel: { CE: 1, EE: Infinity }
  },
  source: {
    createSource: { CE: Infinity, EE: Infinity },
    sourceType: {
      CE: [
        { title: "Image", type: "I", icon: <FiImage size="50" /> },
        { title: "Video", type: "V", icon: <FiVideo size="50" /> },
        { title: "Tabular", type: "T", icon: <FiFileText size="48" /> }
      ],
      EE: [
        { title: "Image", type: "I", icon: <FiImage size="50" /> },
        { title: "Video", type: "V", icon: <FiVideo size="50" /> },
        { title: "RealTime", type: "R", icon: <FiTv size="48" /> },
        { title: "Tabular", type: "T", icon: <FiFileText size="48" /> }
      ]
    },
    schedule: { CE: false, EE: true },
    importModel: { CE: true, EE: false }
  },
  project: {
    createProject: { CE: 1, EE: Infinity }
  }
}

function useEnterpriseDivision(division, type, config) {
  return c[type][config][division]
}

export default useEnterpriseDivision
