import React, { useState } from "react"
import useKeypress from "react-use-keypress"
import { BiSearch } from "react-icons/bi"
const Header2 = props => {
  const [searchKey, setSearchKey] = useState("")
  const [searchFocus, setSearchFocus] = useState(false)
  useKeypress("Enter", () => {
    // Do something when the user has pressed the Escape key
    if (searchFocus) props.onSearch(searchKey)
  })
  return (
    <div className="bluai-main-header">
      <div className="bluai-main-header-title">{props.title}</div>
      {props.buttons?.map((ele, i) => {
        const BtnIcon = ele.btnIcon
        return (
          <div className="bluai-main-header-button hover-pointer" key={i} onClick={ele.onClick} style={{ background: ele.color }}>
            <div className="bluai-main-header-button-icon">
              <BtnIcon />
            </div>
            <div className="bluai-main-header-button-title">{ele.btnTitle}</div>
          </div>
        )
      })}
      {props.onSearch && (
        <div className="bluai-main-header-search">
          <input
            type="text"
            onChange={e => setSearchKey(e.target.value)}
            value={searchKey}
            onFocus={() => setSearchFocus(true)}
            onBlur={() => setSearchFocus(false)}
          />
          <div className="bluai-main-header-search-icon">
            <BiSearch />
          </div>
        </div>
      )}
    </div>
  )
}

export default React.memo(Header2)
