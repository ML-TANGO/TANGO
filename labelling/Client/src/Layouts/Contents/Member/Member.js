import React, { useState, useEffect, useCallback, useRef } from "react"
import { Container } from "reactstrap"
import { confirmAlert } from "react-confirm-alert"
import { toast } from "react-toastify"
import { Transition } from "react-spring/renderprops"

import { BsTrash } from "react-icons/bs"
import { FaPlusCircle } from "react-icons/fa"
import { FiEdit, FiTrash2 } from "react-icons/fi"
import { MdDeleteForever, MdError } from "react-icons/md"

import * as AuthApi from "Config/Services/AuthApi"

import Header2 from "../../../Components/Common/Header2"
import CommonToast from "../../../Components/Common/CommonToast"
import CommonButton from "../../../Components/Common/CommonButton"
import VirtualList from "../../../Components/Common/VirtualList"
import Filter from "../../../Components/Filter/Filter"

import useListHeight from "../../../Components/Utils/useListHeight"
import NewMemberPanel from "./NewMemberPanel"

const sortList = {
  createdSortByNewest: { key: "CRN_DTM", direction: "DESC" },
  createdSortByOldest: { key: "CRN_DTM", direction: "ASC" },
  sortAZ: { key: "IS_TITLE", direction: "DESC" },
  sortZA: { key: "IS_TITLE", direction: "ASC" }
}

const statusList = [
  { items: [{ title: "ROLE", key: "ROLE" }] },
  { items: [{ title: "USE", key: "USE", mapper: { 0: "USE", 1: "USE", 2: "LOCK" } }] }
]

function Member(props) {
  const { history } = props
  const [filterList, setFilterList] = useState([])
  const [searchKey, setSearchKey] = useState(null)
  const [isPanel, setIsPanel] = useState(false)
  const [editData, setEditData] = useState({})
  const [isLoad, setIsLoad] = useState(false)
  const [list, setList] = useState([])
  const listHeight = useListHeight()

  const confirmTextRef = useRef("")

  useEffect(() => {
    initMember()
  }, [])

  const initMember = () => {
    setIsLoad(true)
    AuthApi._getUsers({})
      .then(result => {
        setList(result)
        setFilterList(result)
        setIsLoad(false)
      })
      .catch(e => {
        console.log(e)
        setIsLoad(false)
      })
  }

  useEffect(() => {
    if (searchKey !== null) findItem(searchKey)
  }, [searchKey])

  const panelToggle = () => {
    setIsPanel(isPanel => !isPanel)
  }

  const _handleRemove = useCallback(
    data => () => {
      confirmAlert({
        customUI: ({ onClose }) => {
          return (
            <div className="react-confirm-alert-custom">
              <h1>
                <BsTrash />
                Delete Member
              </h1>
              <div className="custom-modal-body">
                <div className="text-warning">Warning. This action is irreversible.</div>
                <div className="explain">
                  Remove member [ Member ID : <strong>{data.USER_ID}</strong> ].
                </div>
                <div>
                  Please enter <strong>[ {data.USER_ID} ]</strong> to remove.
                </div>
                <input
                  type="text"
                  className="react-confirm-alert-input"
                  onChange={e => {
                    confirmTextRef.current = e.target.value
                  }}
                />
              </div>
              <div className="custom-buttons">
                <CommonButton
                  className="bg-green"
                  text="Apply"
                  onClick={() => {
                    if (confirmTextRef.current.trim() === data.USER_ID.trim()) {
                      document.getElementById("wrapper").scrollIntoView()
                      onClose()
                      AuthApi._deleteUser({ USER_ID: data.USER_ID })
                        .then(result => {
                          if (result.status === 1) {
                            const filter = list.filter(ele => {
                              return ele.USER_ID !== data.USER_ID
                            })
                            setList(filter)
                            const filter2 = filterList.filter(ele => {
                              return ele.USER_ID !== data.USER_ID
                            })
                            setFilterList(filter2)
                            toast.info(<CommonToast Icon={MdDeleteForever} text={"User Delete Success"} />)
                          } else {
                            throw { err: "status 0" }
                          }
                        })
                        .catch(e => {
                          toast.error(<CommonToast Icon={MdError} text={"User Delete Fail"} />)
                          console.log(e)
                        })
                    } else alert("Not matched.")
                  }}
                />
                <CommonButton className="bg-red" text="Cancel" onClick={onClose} />
              </div>
            </div>
          )
        }
      })
    },
    [list, filterList]
  )

  const _handleEdit = useCallback(
    data => () => {
      setIsPanel(isPanel => !isPanel)
      setEditData({ dataInfo: data, pageMode: "EDIT" })
    },
    [history]
  )

  const getCardFunc = data => {
    let arr = [
      {
        func: _handleEdit(data),
        label: "Edit",
        icon: <FiEdit />
      },
      {
        func: _handleRemove(data),
        label: "Delete",
        icon: <FiTrash2 />
      }
    ]
    return arr
  }

  const findItem = value => {
    let newList = list.filter(member => member.USER_ID.indexOf(value) !== -1 || String(member.USER_NM).indexOf(value) !== -1)
    setFilterList(newList)
  }

  return (
    <Container style={{ marginBottom: "3rem" }}>
      <Transition
        unique
        reset
        items={isPanel}
        from={{ transform: "translate3d(100%,0,0)" }}
        enter={{ transform: "translate3d(0%,0,0)" }}
        leave={{ transform: "translate3d(100%,0,0)" }}
      >
        {isPanel =>
          isPanel &&
          (springProps => (
            <NewMemberPanel springProps={springProps} panelToggle={panelToggle} editData={editData} initMember={initMember} />
          ))
        }
      </Transition>
      <Header2
        title="Member"
        buttons={[
          {
            btnTitle: "Add New Member",
            color: "rgb(98, 181, 86)",
            btnIcon: FaPlusCircle,
            btnClass: "bg-blue",
            onClick: () => {
              setEditData({ dataInfo: {}, pageMode: "NEW" })
              panelToggle()
            }
          }
        ]}
        onSearch={setSearchKey}
      />

      <Filter
        filterType={"M"}
        showFilter={false}
        list={list}
        filterList={filterList}
        sortList={sortList}
        setList={setFilterList}
        listCount={filterList.length}
      />
      <VirtualList
        data={filterList}
        height={listHeight}
        funcList={getCardFunc}
        searchKey={searchKey}
        type="M"
        status={statusList}
        isLoad={isLoad}
      />
    </Container>
  )
}

Member.propTypes = {}

export default Member
