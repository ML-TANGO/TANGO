import React, { useState, useEffect } from "react"
import { Input } from "reactstrap"
import { Controller } from "react-hook-form"
import LoadingOverlay from "react-loading-overlay"
import SyntaxHighlighter from "react-syntax-highlighter/dist/esm/default-highlight"
import vs2015 from "react-syntax-highlighter/dist/esm/styles/hljs/vs2015"

import CommonSelect from "../Common/CommonSelect"
import DropZone from "../Common/CommonDropZone"

const definedOptions = {
  DATA_TYPE: [
    { value: "I", label: "Image" },
    { value: "V", label: "Video" },
    { value: "T", label: "Text" }
  ],
  I_OBJECT_TYPE: [
    { value: "D", label: "Detection" },
    { value: "C", label: "Classification" },
    { value: "S", label: "Segmentation" }
  ],
  V_OBJECT_TYPE: [
    { value: "D", label: "Detection" },
    { value: "S", label: "Segmentation" }
  ],
  T_OBJECT_TYPE: [
    { value: "C", label: "Classification" },
    { value: "R", label: "Regressor" }
  ],
  AUTO_TYPE: [
    { value: "N", label: "NO" },
    { value: "Y", label: "YES" }
  ],
  UPLOAD_TYPE: [
    { value: "FILE", label: "FILE" },
    { value: "DB", label: "DATABASE" }
  ],
  DB_TYPE: [
    { value: "oracledb", label: "ORACLE" },
    { value: "mysql", label: "MYSQL" },
    { value: "pg", label: "PostgreSQL" }
  ],
  DATASET_FILTER: [
    {
      label: "ALL",
      options: [
        { value: "allDatasets", label: "All Datasets" },
        { value: "auto", label: "Auto labeling" }
      ]
    },
    {
      label: "OBJECT TYPE",
      options: [
        { value: "classification", label: "Classification" },
        { value: "detection", label: "Detection" },
        { value: "segmentaton", label: "Segmentaton" },
        { value: "regression", label: "Regression" }
      ]
    },
    {
      label: "DATA TYPE",
      options: [
        { value: "image", label: "Image" },
        { value: "video", label: "Video" },
        { value: "tabular", label: "Tabular" }
      ]
    },
    {
      label: "STATUS",
      options: [
        { value: "ready", label: "Ready" },
        { value: "creating", label: "Creating" },
        { value: "failed", label: "Failed" }
      ]
    }
  ],
  AI_FILTER: [
    {
      label: "ALL",
      options: [{ value: "allDatasets", label: "All Trainer" }]
    },
    {
      label: "OBJECT TYPE",
      options: [
        { value: "classification", label: "Classification" },
        { value: "detection", label: "Detection" },
        { value: "segmentaton", label: "Segmentaton" },
        { value: "regression", label: "Regression" }
      ]
    },
    {
      label: "DATA TYPE",
      options: [
        { value: "image", label: "Image" },
        { value: "video", label: "Video" },
        { value: "tabular", label: "Tabular" }
      ]
    },
    {
      label: "STATUS",
      options: [
        { value: "created", label: "Ready" },
        { value: "ready", label: "Ready to Train" },
        { value: "learn", label: "Training" },
        { value: "stop", label: "Stop Training" },
        { value: "trained", label: "Trained" },
        { value: "failed", label: "Failed" }
      ]
    }
  ],
  DATASET_SORT: [
    {
      label: "CREATED",
      options: [
        { value: "createdSortByNewest", label: "Sort by newest" },
        { value: "createdSortByOldest", label: "Sort by oldest" }
      ]
    },
    {
      label: "UPDATED",
      options: [
        { value: "updatedSortByNewest", label: "Sort by newest" },
        { value: "updatedSortByOldest", label: "Sort by oldest" }
      ]
    },

    {
      label: "TITLE",
      options: [
        { value: "sortAZ", label: "Sort A-Z" },
        { value: "sortZA", label: "Sort Z-A" }
      ]
    },
    {
      label: "FILE",
      options: [
        { value: "sortFileCountASC", label: "File count ASC" },
        { value: "sortFileCountDESC", label: "File count DESC" },
        { value: "sortFileSizeASC", label: "File size ASC" },
        { value: "sortFileSizeDESC", label: "File size DESC" }
      ]
    }
  ],
  AI_SORT: [
    {
      label: "CREATED",
      options: [
        { value: "createdSortByNewest", label: "Sort by newest" },
        { value: "createdSortByOldest", label: "Sort by oldest" }
      ]
    },
    {
      label: "UPDATED",
      options: [
        { value: "updatedSortByNewest", label: "Sort by newest" },
        { value: "updatedSortByOldest", label: "Sort by oldest" }
      ]
    },

    {
      label: "TITLE",
      options: [
        { value: "sortAZ", label: "Sort A-Z" },
        { value: "sortZA", label: "Sort Z-A" }
      ]
    },
    {
      label: "MODEL",
      options: [
        { value: "sortModelCountASC", label: "Model count ASC" },
        { value: "sortModelCountDESC", label: "Model count DESC" },
        { value: "sortModelSizeASC", label: "Model size ASC" },
        { value: "sortModelSizeDESC", label: "Model size DESC" },
        { value: "sortTrainTimeASC", label: "Training time ASC" },
        { value: "sortTrainTimeDESC", label: "Training time DESC" }
      ]
    }
  ],
  IS_FILTER: [
    {
      label: "ALL",
      options: [
        { value: "allDatasets", label: "All Service" },
        { value: "active", label: "Active Service" }
      ]
    },
    {
      label: "OBJECT TYPE",
      options: [
        { value: "classification", label: "Classification" },
        { value: "detection", label: "Detection" },
        { value: "segmentaton", label: "Segmentaton" },
        { value: "regression", label: "Regression" }
      ]
    },
    {
      label: "SERVICE TYPE",
      options: [
        { value: "realtime", label: "Realtime" },
        { value: "image", label: "Image" },
        { value: "video", label: "Video" },
        { value: "tabular", label: "Tabular" }
      ]
    },
    {
      label: "STATUS",
      options: [
        { value: "created", label: "Created" },
        { value: "active", label: "Active" },
        { value: "stopped", label: "Stopped" },
        { value: "failed", label: "Failed" }
      ]
    }
  ],
  IS_SORT: [
    {
      label: "CREATED",
      options: [
        { value: "createdSortByNewest", label: "Sort by newest" },
        { value: "createdSortByOldest", label: "Sort by oldest" }
      ]
    },
    {
      label: "UPDATED",
      options: [
        { value: "updatedSortByNewest", label: "Sort by newest" },
        { value: "updatedSortByOldest", label: "Sort by oldest" }
      ]
    },

    {
      label: "TITLE",
      options: [
        { value: "sortAZ", label: "Sort A-Z" },
        { value: "sortZA", label: "Sort Z-A" }
      ]
    }
  ],
  QP_FILTER: [
    {
      label: "ALL",
      options: [{ value: "allDatasets", label: "All Analytics" }]
    }
    // {
    //   label: "OBJECT TYPE",
    //   options: [
    //     { value: "classification", label: "Classification" },
    //     { value: "detection", label: "Detection" },
    //     { value: "segmentaton", label: "Segmentaton" },
    //     { value: "regression", label: "Regression" }
    //   ]
    // },
    // {
    //   label: "SERVICE TYPE",
    //   options: [
    //     { value: "realtime", label: "Realtime" },
    //     { value: "image", label: "Image" },
    //     { value: "video", label: "Video" },
    //     { value: "tabular", label: "Tabular" }
    //   ]
    // }
  ],
  QP_SORT: [
    {
      label: "CREATED",
      options: [
        { value: "createdSortByNewest", label: "Sort by newest" },
        { value: "createdSortByOldest", label: "Sort by oldest" }
      ]
    },
    {
      label: "UPDATED",
      options: [
        { value: "updatedSortByNewest", label: "Sort by newest" },
        { value: "updatedSortByOldest", label: "Sort by oldest" }
      ]
    },
    {
      label: "TITLE",
      options: [
        { value: "sortAZ", label: "Sort A-Z" },
        { value: "sortZA", label: "Sort Z-A" }
      ]
    }
  ]
}

export const FormTitle = React.memo(({ title, titleClassName }) => {
  return (
    <span className={`form__form-group-label ${titleClassName}`} style={{ width: "20%", textAlign: "right" }}>
      {title}
    </span>
  )
})

export const FormText = React.memo(({ title, titleClassName, name, register, errors, disabled, defaultValue, type }) => {
  return (
    <>
      <div className="d-flex w-100 mt-1">
        <FormTitle title={title} titleClassName={titleClassName} />
        <Input
          type={type}
          autoComplete="off"
          className="mt-1"
          id={name}
          name={name}
          innerRef={register}
          style={{ width: "80%" }}
          disabled={disabled}
          defaultValue={defaultValue}
        />
      </div>
      {errors && errors[name]?.type === "required" && <div className="form__form-group-label form-error mt-1">{title} is required</div>}
      {errors && errors[name]?.type === "validateTrim" && <div className="form__form-group-label form-error mt-1">{title} is required</div>}
      {errors && errors[name]?.type === "validateLength" && (
        <div className="form__form-group-label form-error mt-1">
          The maximum number of characters that can be generated has been exceeded
        </div>
      )}
      {errors && errors[name]?.type === "validateIdCheck" && (
        <div className="form__form-group-label form-error mt-1">This is a duplicate ID</div>
      )}
    </>
  )
})

FormText.defaultProps = {
  disabled: false,
  type: "text"
}

export const FormPassword = React.memo(({ title, titleClassName, name, register, errors, disabled, defaultValue }) => {
  return (
    <>
      <div className="d-flex w-100 mt-1">
        <FormTitle title={title} titleClassName={titleClassName} />
        <Input
          type="password"
          autoComplete="off"
          className="mt-1"
          id={name}
          name={name}
          innerRef={register}
          style={{ width: "80%" }}
          disabled={disabled}
          defaultValue={defaultValue}
        />
      </div>
      {errors && errors[name]?.type === "required" && <div className="form__form-group-label form-error mt-1">{title} is required</div>}
      {errors && errors[name]?.type === "validateTrim" && <div className="form__form-group-label form-error mt-1">{title} is required</div>}
    </>
  )
})

FormText.defaultProps = {
  disabled: false
}

export const FormTextArea = React.memo(({ title, titleClassName, register, name, height, errors }) => {
  return (
    <>
      <div className="d-flex w-100 mt-1">
        <FormTitle title={title} titleClassName={titleClassName} />
        <Input type="textarea" name={name} className="mt-1" innerRef={register} style={{ resize: "none", width: "80%", height: height }} />
      </div>
      {errors && errors[name]?.type === "required" && <div className="form__form-group-label form-error mt-1">{title} is required</div>}
      {errors && errors[name]?.type === "validateTrim" && <div className="form__form-group-label form-error mt-1">{title} is required</div>}
    </>
  )
})

export const FormNumber = React.memo(({ title, titleClassName, name, register, disabled, placeholder, defaultValue, type }) => {
  return (
    <div className="d-flex w-100 mt-1">
      <FormTitle title={title} titleClassName={titleClassName} />
      <Input
        type="number"
        className="mt-1"
        id={name}
        name={name}
        innerRef={register}
        placeholder={placeholder}
        style={{ width: "80%" }}
        onKeyPress={e => {
          if (type !== "FLOAT") {
            switch (e.key) {
              case "1":
              case "2":
              case "3":
              case "4":
              case "5":
              case "6":
              case "7":
              case "8":
              case "9":
              case "0":
                e.returnValue = true
                break
              default:
                e.preventDefault()
                e.returnValue = false
                break
            }
          }
        }}
        disabled={disabled}
        defaultValue={defaultValue}
      />
    </div>
  )
})

FormNumber.defaultProps = {}

export const FormNoneInput = React.memo(({ title, titleClassName, name, defaultValue, register, setValue, watch }) => {
  const [check, setCheck] = useState(true)
  const watchValue = watch(name, defaultValue)

  const handleCheck = e => {
    setCheck(e.target.checked)
    if (e.target.checked) {
      setValue(name, defaultValue)
    } else {
      setValue(name, "")
    }
  }

  useEffect(() => {
    if (watchValue !== defaultValue && check) {
      setCheck(false)
    }
  }, [watchValue])

  return (
    <div className="d-flex w-100 mt-1">
      <FormTitle title={title} titleClassName={titleClassName} />
      <div className="d-flex" style={{ width: "80%" }}>
        <div className="h-100 d-flex">
          <input
            className="w-auto h-auto mr-2 align-self-center"
            type="checkbox"
            name="none-select"
            checked={check}
            onClick={handleCheck}
            readOnly
          />
          <span className="mr-2 align-self-center">{defaultValue}</span>
        </div>
        <Input
          type="text"
          className="mt-1"
          id={name}
          name={name}
          innerRef={register}
          style={{ borderRadius: "10px !important" }}
          onKeyPress={e => {
            switch (e.key) {
              case "1":
              case "2":
              case "3":
              case "4":
              case "5":
              case "6":
              case "7":
              case "8":
              case "9":
              case "0":
                e.returnValue = true
                break
              case ".":
                if (/^\d*[.]\d*$/.test(e.target.value + ".")) {
                  e.returnValue = true
                } else {
                  e.preventDefault()
                  e.returnValue = false
                }
                break
              default:
                e.preventDefault()
                e.returnValue = false
                break
            }
          }}
          disabled={check}
          defaultValue={defaultValue}
        />
      </div>
    </div>
  )
})

export const FormSelect = React.memo(
  ({
    title,
    name,
    onChange,
    control,
    disabled,
    dataType,
    options,
    defaultValue,
    rules,
    inputRef,
    onFocus,
    isDefault,
    isMulti,
    placeholder,
    isClearable,
    prefix,
    group,
    titleClassName,
    menuPortalTarget
  }) => {
    return (
      <div className="d-flex w-100 mt-1">
        <FormTitle title={title} titleClassName={titleClassName} />
        <Controller
          inputRef={inputRef}
          style={{ width: "80%" }}
          className="mt-1"
          placeholder={placeholder}
          as={CommonSelect}
          name={name}
          valueName="selected"
          control={control}
          options={options ? options : dataType ? definedOptions[dataType + "_" + name] : definedOptions[name]}
          onChange={onChange}
          isMulti={isMulti}
          defaultValue={
            isDefault
              ? defaultValue
                ? defaultValue
                : options
                ? options[0]?.value
                : dataType
                ? definedOptions[dataType + "_" + name][0].value
                : group
                ? definedOptions[name][0].options[0].value
                : definedOptions[name][0].value
              : undefined
          }
          isClearable={isClearable}
          isDefault={isDefault}
          disabled={disabled}
          rules={rules}
          onFocus={onFocus}
          prefix={prefix}
          group={group}
          menuPortalTarget={menuPortalTarget}
        />
      </div>
    )
  }
)

FormSelect.defaultProps = {
  disabled: false,
  isMulti: false
}

export const FormDropZone = React.memo(({ control, typeState, pageState, setFileList, accept, multiple, onChange, maxFiles }) => {
  const _dropzoneChange = ([selected]) => {
    setFileList(selected)
    return selected
  }

  return (
    <>
      <LoadingOverlay
        active={pageState.isUpload || pageState.pageMode == "DUPLICATION"}
        spinner={pageState.isUpload}
        text={pageState.pageMode == "DUPLICATION" ? "DUPLICATION can't Upload" : "Uploading..."}
      >
        <Controller
          as={DropZone}
          control={control}
          name="files"
          onChange={onChange ? onChange : _dropzoneChange}
          height={"270px"}
          accept={
            accept
              ? accept
              : typeState.dataType === "I"
              ? "image/jpeg, image/png"
              : typeState.dataType === "V"
              ? "video/mp4, video/webm, video/ogg"
              : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, text/plain, text/csv, application/vnd.ms-excel"
          }
          maxFiles={maxFiles}
          defaultValue={[]}
          multiple={multiple}
        />
      </LoadingOverlay>
    </>
  )
})

FormDropZone.defaultProps = {
  multiple: true
}

export const FormIcon = React.memo(
  ({ title, titleClassName, iconList, name, register, setValue, watch, setTypeState, typeName, defaultValue, typeState, onChange }) => {
    const watchValue = watch(name, defaultValue)

    useEffect(() => {
      register(name)
      setValue(name, defaultValue)
    }, [])

    const onClick = type => () => {
      setValue(name, type)
      if (onChange) onChange(type)
    }

    useEffect(() => {
      setTypeState(prevState => ({ ...prevState, [typeName]: watchValue }))
    }, [watchValue])

    return (
      <div className="d-flex w-100 mt-1">
        <FormTitle title={title} titleClassName={titleClassName} />
        <div style={{ width: "80%" }} className="d-flex">
          {iconList.map((el, i) =>
            el?.viewType?.includes(typeState.dataType) === false ? (
              <div key={i} className={"mr-4 mt-1"} style={{ width: "20%" }}>
                <div
                  className={`d-flex justify-content-center hover-pointer ${watchValue === el.type ? "form-icon-active" : ""}`}
                  style={{ border: "2px solid #33333a", borderRadius: "10px", height: "90px", alignItems: "center", color: "#333" }}
                >
                  {el.icon}
                </div>
                <div className="mt-1 mb-1" style={{ textAlign: "center", lineHeight: 1.2 }}>
                  {el.title}
                </div>
              </div>
            ) : (
              <div key={i} className={"mr-4 mt-1"} style={{ width: "20%" }}>
                <div
                  className={`d-flex justify-content-center hover-pointer ${watchValue === el.type ? "form-icon-active" : ""}`}
                  style={{ border: "2px solid #33333a", borderRadius: "10px", height: "90px", alignItems: "center" }}
                  onClick={onClick(el.type)}
                >
                  {el.icon}
                </div>
                <div className="mt-1 mb-1" style={{ textAlign: "center", lineHeight: 1.2 }}>
                  {el.title}
                </div>
              </div>
            )
          )}
        </div>
      </div>
    )
  }
)

export const FormEditor = ({ title, titleClassName, register, name, height, errors, watch, getValues }) => {
  const watchValue = watch(name, getValues(name))

  return (
    <>
      <div className="d-flex w-100 mt-1">
        <FormTitle title={title} titleClassName={titleClassName} />
        <div className="d-flex" style={{ width: "80%", flexDirection: "column" }}>
          <Input
            type="textarea"
            name={name}
            className="mb-2"
            innerRef={register}
            style={{ resize: "none", width: "100%", height: height }}
          />
          <SyntaxHighlighter
            language="sql"
            style={vs2015}
            customStyle={{ height: height, width: "100%", textAlign: "left", borderRadius: "10px" }}
          >
            {watchValue}
          </SyntaxHighlighter>
        </div>
      </div>
      {errors && errors[name]?.type === "required" && (
        <div className="form__form-group-label form-error mt-1 text-left">{title} is required</div>
      )}
    </>
  )
}
