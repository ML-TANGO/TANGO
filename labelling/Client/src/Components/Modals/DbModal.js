import React, { useState } from "react"
import { Row, Col, Modal, ModalHeader, ModalBody, ModalFooter } from "reactstrap"
import { toast } from "react-toastify"
import { FaDatabase } from "react-icons/fa"
import { FormText, FormSelect, FormPassword, FormNumber, FormEditor } from "../Form/FormComponent"
import { FaCheckCircle, FaRunning } from "react-icons/fa"
import { MdError } from "react-icons/md"
import { makeStyles } from "@material-ui/core/styles"
import CircularProgress from "@material-ui/core/CircularProgress"
import { useForm } from "react-hook-form"

//apis
import * as TabDataSetApi from "../../Config/Services/TabDataSetApi"

// custom components
import CommonButton from "../Common/CommonButton"
import CommonToast from "../Common/CommonToast"
import VirtualTable from "../Common/VirtualTable"

const useStyles = makeStyles(theme => ({
  root: {
    display: "flex",
    alignItems: "center",
    marginRight: "auto !important",
    marginLeft: "10px"
  },
  wrapper: {
    margin: theme.spacing(1),
    position: "relative"
  },
  buttonProgress: {
    color: "green",
    position: "absolute",
    top: "50%",
    left: "50%",
    marginTop: -12,
    marginLeft: -12,
    zIndex: 999
  }
}))

function DbModal(props) {
  const { modal, toggle, setTableState, tableState } = props
  const [loading, setLoading] = useState(false)
  const [columns, setColumns] = useState([])
  const [tableData, setTableData] = useState([])
  const classes = useStyles()
  const { register, handleSubmit, control, errors, getValues, watch } = useForm({
    defaultValues: {
      DB_TYPE: tableState.tableInfo.CLIENT ? tableState.tableInfo.CLIENT : "oracledb",
      DB_HOST: tableState.tableInfo.ADDRESS ? tableState.tableInfo.ADDRESS : "",
      DB_PORT: tableState.tableInfo.PORT ? tableState.tableInfo.PORT : "",
      DB_NM: tableState.tableInfo.DBNAME ? tableState.tableInfo.DBNAME : "",
      DB_USER: tableState.tableInfo.USER ? tableState.tableInfo.USER : "",
      DB_PW: tableState.tableInfo.PASSWORD ? tableState.tableInfo.PASSWORD : "",
      DB_QUERY: tableState.tableInfo.QUERY ? tableState.tableInfo.QUERY : "",
      QUERY_LIMIT: tableState.tableInfo.LIMIT ? tableState.tableInfo.LIMIT : 5
      // DB_TYPE: tableState.tableInfo.CLIENT ? tableState.tableInfo.CLIENT : "mysql",
      // DB_HOST: tableState.tableInfo.ADDRESS ? tableState.tableInfo.ADDRESS : "www.wedalab.com",
      // DB_PORT: tableState.tableInfo.PORT ? tableState.tableInfo.PORT : 9156,
      // DB_NM: tableState.tableInfo.DBNAME ? tableState.tableInfo.DBNAME : "BLUAI_AI",
      // DB_USER: tableState.tableInfo.USER ? tableState.tableInfo.USER : "bluai",
      // DB_PW: tableState.tableInfo.PASSWORD ? tableState.tableInfo.PASSWORD : "WEDA_BLUAI_0717",
      // DB_QUERY: tableState.tableInfo.QUERY ? tableState.tableInfo.QUERY : "select * from BLUAI_AI.TAB_DBINFO",
      // QUERY_LIMIT: tableState.tableInfo.LIMIT ? tableState.tableInfo.LIMIT : 10
    }
  })

  const handleData = type => async () => {
    try {
      type === "test" && setLoading(true)
      if (!loading) {
        const dbInfo = getValues()
        const param = {
          IS_TEST: type !== "test" ? false : true,
          CLIENT: dbInfo.DB_TYPE,
          ADDRESS: dbInfo.DB_HOST,
          PORT: Number(dbInfo.DB_PORT),
          DBNAME: dbInfo.DB_NM,
          USER: dbInfo.DB_USER,
          PASSWORD: dbInfo.DB_PW,
          QUERY: dbInfo.DB_QUERY,
          LIMIT: Number(dbInfo.QUERY_LIMIT)
        }

        const dbConn = await TabDataSetApi._getDbConnectionInfo(param)

        if (dbConn.STATUS === 1) {
          if (type === "test") {
            setLoading(false)
            toast.info(<CommonToast Icon={FaCheckCircle} text={"Database connection was successful."} />)
          } else if (type === "run") {
            if (dbConn.DATA.length !== 0) {
              const colInfo = Object.keys(dbConn?.DATA[0])
              const columns = colInfo.map(el => ({
                label: el,
                dataKey: el,
                className: "text-center",
                disableSort: true,
                width: 120
              }))
              const indexColumn = {
                label: "#",
                dataKey: "-",
                className: "text-center",
                disableSort: true,
                width: 80,
                cellRenderer: ({ rowIndex }) => rowIndex + 1
              }
              setColumns([indexColumn, ...columns])
              setTableData(dbConn?.DATA)
              toast.info(<CommonToast Icon={FaCheckCircle} text={"Data query was successful."} />)
            } else {
              toast.info(<CommonToast Icon={MdError} text={"There are no query results."} />)
              setLoading(false)
            }
          } else {
            if (dbConn.DATA.length !== 0) {
              const colInfo = Object.keys(dbConn?.DATA[0])
              const colList = colInfo.map(el => ({ COLUMN_NM: el, DEFAULT_VALUE: "null" }))
              const columns = colInfo.map(el => ({
                label: el,
                dataKey: el,
                className: "text-center",
                disableSort: true,
                width: 120
              }))
              const indexColumn = {
                label: "#",
                dataKey: "-",
                className: "text-center",
                disableSort: true,
                width: 80,
                cellRenderer: ({ rowIndex }) => rowIndex + 1
              }
              setTableState(prevState => ({
                ...prevState,
                tableData: dbConn.DATA,
                tableInfo: param,
                colList: colList,
                columns: [indexColumn, ...columns]
              }))
              toast.info(<CommonToast Icon={FaCheckCircle} text={"Data upload was successful."} />)
              toggle()
            } else {
              toast.info(<CommonToast Icon={MdError} text={"There are no query results."} />)
              setLoading(false)
            }
          }
        } else {
          setLoading(false)
          if (type === "test") {
            toast.error(<CommonToast Icon={MdError} text={"Database connection failed."} />)
          } else if (type === "run") {
            toast.error(<CommonToast Icon={MdError} text={"The data query failed."} />)
          } else {
            toast.error(<CommonToast Icon={MdError} text={"Data upload failed."} />)
          }
        }
      }
    } catch (e) {
      toast.error(<CommonToast Icon={MdError} text={"Server connection failed."} />)
      setLoading(false)
      console.log(e)
    }
  }

  return (
    <Modal
      isOpen={modal}
      toggle={toggle}
      className={"modal-dialog--primary modal-dialog--header"}
      style={{ marginTop: "10rem" }}
      size={"xl"}
    >
      <ModalHeader className="tag-modal-haeder" toggle={toggle}>
        <div className="modal-icon"></div>
        <div className="modal-name">
          <div>
            <h3>
              <FaDatabase /> Connect to Database
            </h3>
          </div>
        </div>
      </ModalHeader>
      <ModalBody>
        <Row>
          <Col xl={6}>
            <div className="form  pr-2">
              <FormSelect
                title="Database Type"
                titleClassName={"mr-4 mt-2"}
                name="DB_TYPE"
                control={control}
                onChange={([selected]) => {
                  return selected
                }}
                menuPortalTarget={false}
                isDefault={true}
              />
              <FormText
                title="Server Host"
                titleClassName={"mr-4 mt-2"}
                control={control}
                name="DB_HOST"
                register={register({
                  required: true,
                  validate: {
                    validateTrim: value => String(value).trim().length !== 0
                  }
                })}
                errors={errors}
              />
              <FormText
                title="Port"
                titleClassName={"mr-4 mt-2"}
                control={control}
                name="DB_PORT"
                register={register({
                  required: true,
                  validate: {
                    validateTrim: value => String(value).trim().length !== 0
                  }
                })}
                errors={errors}
              />
              <FormText
                title="Database Name"
                titleClassName={"mr-4 mt-2"}
                control={control}
                name="DB_NM"
                register={register()}
                errors={errors}
              />
              <FormText
                title="User Name"
                titleClassName={"mr-4 mt-2"}
                name="DB_USER"
                register={register({
                  required: true,
                  validate: {
                    validateTrim: value => String(value).trim().length !== 0
                  }
                })}
                errors={errors}
              />
              <FormPassword
                title="Password"
                titleClassName={"mr-4 mt-2"}
                control={control}
                name="DB_PW"
                register={register()}
                errors={errors}
              />
            </div>
          </Col>
          <Col xl={6}>
            <div className="form  pr-2">
              {/* <FormTextArea
                title="Query"
                titleClassName={"mr-4 mt-2"}
                control={control}
                name="DB_QUERY"
                height="192px"
                register={register({
                  required: true,
                  validate: {
                    validateTrim: value => String(value).trim().length !== 0
                  }
                })}
                errors={errors}
              /> */}
              <FormEditor
                title="Query"
                titleClassName={"mr-4 mt-2"}
                name="DB_QUERY"
                register={register({
                  required: true,
                  validate: {
                    validateTrim: value => String(value).trim().length !== 0
                  }
                })}
                getValues={getValues}
                height="115px"
                errors={errors}
                watch={watch}
              />
              <FormNumber
                title="QUERY_LIMIT"
                titleClassName={"mr-4 mt-2"}
                name="QUERY_LIMIT"
                register={register({
                  required: true,
                  validate: {
                    validateTrim: value => String(value).trim().length !== 0
                  }
                })}
                errors={errors}
              />
            </div>
            <div className="right-button mt-2">
              <CommonButton className="bg-green" ButtonIcon={FaRunning} text="Run" onClick={handleSubmit(handleData("run"))} />
            </div>
          </Col>
        </Row>
        <Row className="mt-2">
          <Col xl={12}>
            <VirtualTable
              className="vt-table text-break-word"
              rowClassName="vt-header"
              height={"200px"}
              width={columns.length * 120}
              headerHeight={40}
              rowHeight={50}
              columns={columns}
              data={tableData}
              // scrollIndex={fileState.successCount}
              // onRowMouseOver={_onRowMouseOver("fileHoverIndex")}
              // onRowMouseOut={_onRowMouseOut("fileHoverIndex")}
              // onRowClick={_onRowClick("selectedFiles")}
              // rowStyle={_rowStyle("fileHoverIndex", "selectedFiles")}
              style={{ overflowX: "scroll", overflowY: "hidden" }}
            />
          </Col>
        </Row>
      </ModalBody>
      <ModalFooter>
        <div className={classes.root}>
          <div className={classes.wrapper}>
            <CommonButton id="test" className="bg-blue" text="Test Connection" onClick={handleData("test")} disabled={loading} />
            {loading && <CircularProgress size={24} className={classes.buttonProgress} />}
          </div>
        </div>
        <CommonButton className="bg-green" ButtonIcon={FaCheckCircle} text="Upload" onClick={handleSubmit(handleData())} />
        <CommonButton className="bg-red" text="Cancel" onClick={toggle} />
      </ModalFooter>
    </Modal>
  )
}

DbModal.propTypes = {}

export default DbModal
