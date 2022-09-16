// react-libraries
import React, { useState, useEffect } from "react"
import { useForm } from "react-hook-form"

// react-icons
import { FiFilter } from "react-icons/fi"
import { BiSortDown } from "react-icons/bi"

// custom components
import { FormSelect } from "Components/Form/FormComponent"

const Filter = ({ filterType, list, listCount, filterList, sortList, setList, showFilter, toggleFilter, setToggleFilter }) => {
  const { control } = useForm()
  const [selectedSort, setSelectedSort] = useState("createdSortByNewest")
  const [filter, setFilter] = useState("allDatasets")

  const sortArray = (array, type, direction) => {
    const sorted =
      direction === "DESC"
        ? [...array].sort((a, b) => {
            return a[type] > b[type] ? -1 : a[type] < b[type] ? 1 : 0
          })
        : [...array].sort((a, b) => {
            return a[type] < b[type] ? -1 : a[type] > b[type] ? 1 : 0
          })
    return sorted
  }

  useEffect(() => {
    if (toggleFilter) {
      let sorted = changeFilteredList(filter)
      const l = changeSortedList(selectedSort, sorted)
      setList(l)
      setToggleFilter(null)
    }
  }, [toggleFilter])

  const changeFilteredList = selected => {
    let type
    let filter
    switch (selected) {
      case "allDatasets":
        filter = list
        break
      case "auto":
        filter = list.filter(ele => ele.AUTO_MODEL !== null)
        break
      case "classification":
        type = "C"
        filter = list.filter(ele => ele.OBJECT_TYPE === type)
        break
      case "detection":
        type = "D"
        filter = list.filter(ele => ele.OBJECT_TYPE === type)
        break
      case "segmentaton":
        type = "S"
        filter = list.filter(ele => ele.OBJECT_TYPE === type)
        break
      case "regression":
        type = "R"
        filter = list.filter(ele => ele.OBJECT_TYPE === type)
        break
      case "image":
        type = "I"
        filter = list.filter(ele => ele.DATA_TYPE === type || ele.IS_TYPE === type)
        break
      case "video":
        type = "V"
        filter = list.filter(ele => ele.DATA_TYPE === type || ele.IS_TYPE === type)
        break
      case "realtime":
        type = "R"
        filter = list.filter(ele => ele.IS_TYPE === type)
        break
      case "tabular":
        type = "T"
        filter = list.filter(ele => ele.DATA_TYPE === type || ele.IS_TYPE === type)
        break
      case "ready":
        filter = list.filter(ele => ele[filterType + "_STS"] === "READY")
        break
      case "learn":
        filter = list.filter(ele => ele[filterType + "_STS"] === "LEARN")
        break
      case "stop":
        filter = list.filter(ele => ele[filterType + "_STS"] === "STOP")
        break
      case "creating":
        filter = list.filter(ele => ele[filterType + "_STS"] === "CREATE")
        break
      case "trained":
        filter = list.filter(ele => ele[filterType + "_STS"] === "DONE")
        break
      case "created":
        filter = list.filter(ele => ele[filterType + "_STS"] === "NONE")
        break
      case "stopped":
        filter = list.filter(ele => ele[filterType + "_STS"] === "DONE")
        break
      case "active":
        filter = list.filter(ele => ele[filterType + "_STS"] === "ACTIVE")
        break
      case "failed":
        filter = list.filter(
          ele =>
            ele[filterType + "_STS"] === "CRN_FAIL" ||
            ele[filterType + "_STS"] === "AUTO_FAIL" ||
            ele[filterType + "_STS"] === "FAIL" ||
            ele[filterType + "_STS"] === "ACT_FAIL"
        )
        break
      default:
        filter = list
        break
    }
    let sorted = sortArray(filter, sortList[selectedSort].key, sortList[selectedSort].direction)
    return sorted
  }

  const changeSortedList = (selected, list) => {
    setSelectedSort(selected)
    let sorted = sortArray(list, sortList[selected].key, sortList[selected].direction)
    return sorted
  }

  return (
    <div className="ds-list-control">
      <div className="ds-list-control-count">COUNT: {listCount}</div>
      {showFilter && (
        <>
          <div className="ds-list-control-filter">
            <FormSelect
              title={<FiFilter />}
              titleClassName={"mb-1 mt-1"}
              name={filterType + "_FILTER"}
              control={control}
              onChange={([selected]) => {
                setFilter(selected)
                const sorted = changeFilteredList(selected)
                setList(sorted)
                return selected
              }}
              isDefault={true}
              group={true}
            />
          </div>
          <div className="ds-list-control-sort">
            <FormSelect
              title={<BiSortDown />}
              titleClassName={"mb-1 mt-1"}
              name={filterType + "_SORT"}
              control={control}
              onChange={([selected]) => {
                const sorted = changeSortedList(selected, filterList)
                setList(sorted)
                return selected
              }}
              isDefault={true}
              group={true}
            />
          </div>
        </>
      )}
    </div>
  )
}

Filter.defaultProps = {
  showFilter: true
}

export default Filter
