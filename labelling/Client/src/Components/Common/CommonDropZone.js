import React, { useCallback, useState } from "react"
import PropTypes from "prop-types"
import { useDropzone } from "react-dropzone"
import styled from "styled-components"
import path from "path"
import { toast } from "react-toastify"
import { MdError } from "react-icons/md"
import { FaUpload } from "react-icons/fa"
import CommonToast from "./CommonToast"

const getColor = props => {
  if (props.isDragAccept) {
    return "#00e676"
  }
  if (props.isDragReject) {
    return "#ff1744"
  }
  return "#888888"
}

const getBackgroundColor = props => {
  if (props.disabled) {
    return "#343a40"
  } else {
    return "#000"
  }
}

const Container = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  border-width: 2px;
  border-radius: 5px;
  border-color: ${props => getColor(props)};
  border-style: dashed;
  background-color: ${props => getBackgroundColor(props)};
  color: #bdbdbd;
  outline: none;
  transition: border 0.24s ease-in-out;
  height: 100%;
  justify-content: center;
`
const strByteLength = function (s, b, i, c) {
  for (b = i = 0; (c = s.charCodeAt(i++)); b += c >> 11 ? 3 : c >> 7 ? 2 : 1);
  return b
}

function CommonDropZone(props) {
  const onDrop = useCallback(
    acceptedFiles => {
      if (acceptedFiles.length <= props.maxFiles) {
        let arr = []

        acceptedFiles.forEach(file => {
          const len = strByteLength(file.name)
          let base = path.basename(path.dirname(file.path))
          file.progress = 0
          file.status = 0
          file.base = base === "." ? "untagged" : base
          if (len <= 256) arr.push(file)
          else
            toast.error(<CommonToast Icon={MdError} text={`File size exceeded 256 bytes \n FileName : ${file.name} `} />, {
              autoClose: 4000
            })
        })
        props.onChange(arr)
      } else {
        toast.error(<CommonToast Icon={MdError} text={`Drag and drop is supported for up to 1000 items`} />, { autoClose: 4000 })
      }
    },
    [props.onChange]
  )

  const { getRootProps, getInputProps, isDragActive, isDragAccept, isDragReject } = useDropzone({
    onDrop,
    accept: props.accept,
    multiple: props.multiple,
    disabled: props.disabled
  })

  return (
    <div
      className="p-2 mt-2 w-100"
      style={{
        backgroundColor: "black",
        height: props.height,
        borderRadius: "10px"
      }}
    >
      <Container {...getRootProps({ isDragActive, isDragAccept, isDragReject })} disabled={props.disabled}>
        <input {...getInputProps()} />
        <div>
          <FaUpload className="mr-2" size="15px" />
          {props.title}
        </div>
      </Container>
    </div>
  )
}

CommonDropZone.propTypes = {
  onChange: PropTypes.func.isRequired,
  height: PropTypes.string,
  accept: PropTypes.string,
  title: PropTypes.string,
  multiple: PropTypes.bool,
  disabled: PropTypes.bool,
  maxFiles: PropTypes.number
}

CommonDropZone.defaultProps = {
  disabled: false,
  multiple: true,
  title: `Drag 'n' drop some files here.`,
  accept: "",
  maxFiles: Infinity
}

export default CommonDropZone
