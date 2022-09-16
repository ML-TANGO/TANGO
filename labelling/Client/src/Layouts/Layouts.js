/* eslint-disable react/prop-types */
import React, { useEffect, Suspense } from "react"
import { Route, Switch, useHistory } from "react-router-dom"
import { useSelector, useDispatch } from "react-redux"
import ReactTooltip from "react-tooltip"
import { toast } from "react-toastify"
import { MdError } from "react-icons/md"
import BeatLoader from "react-spinners/BeatLoader"
import { css } from "@emotion/core"

import CommonToast from "../Components/Common/CommonToast"
import MainWrapper from "Components/MainWrapper"

const DashBoard = React.lazy(() => import(/* webpackPrefetch: true */ "./Contents/DashBoard/DashBoard"))
const DataSets = React.lazy(() => import(/* webpackPrefetch: true */ "./Contents/DataSets/DataSets"))
const SideBar = React.lazy(() => import(/* webpackPrefetch: true */ "./SideBar/SideBar"))
const Member = React.lazy(() => import(/* webpackPrefetch: true */ "./Contents/Member/Member"))

const DataSetAnalytics = React.lazy(() => import("./Contents/DataSets/DataSetsAnalytics/DataSetAnalytics"))
const LabelLayout = React.lazy(() => import("./Contents/DataSets/Labelling/LabelLayout"))

import AxiosInterceptors from "../Components/Utils/AxiosInterceptors"

import * as CommonActions from "Redux/Actions/CommonActions"

const override = css`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
`

const NavRoute = ({ exact, path, component: Component }) => <Route exact={exact} path={path} render={props => <Component {...props} />} />

const Layouts = () => {
  // const [collapse, setCollapse] = useState(false)
  const history = useHistory()
  const dispatch = useDispatch()
  // AxiosInterceptors()
  // useEffect(() => {
  //   const userInfo = window.sessionStorage.getItem("userInfo")
  //   if (userInfo === null || userInfo === "" || userInfo === undefined) {
  //     toast.error(<CommonToast Icon={MdError} text={"please try again after logging in"} />)
  //     history.push({ pathname: "/" })
  //   }
  // }, [])

  const commonRedux = useSelector(
    state => state.common,
    (prevItem, item) => {
      if (item === prevItem) return true
      else return false
    }
  )

  useEffect(() => {
    history.listen(() => {
      ReactTooltip.hide()
    })
  }, [history])

  const onCollapse = () => {
    dispatch(CommonActions._collapseSidebar(!commonRedux.collapse))
    // setCollapse(collapse => !collapse)
  }

  return (
    <MainWrapper className={`without-y-scrollbar no-topbar`}>
      {/* {commonRedux.viewTopbar && <TopBar {...props} />} */}
      <SideBar collapse={commonRedux.collapse} onCollapse={onCollapse} />
      <div className={`${commonRedux.collapse ? "sidebar-close" : "sidebar-open"}`}>
        <Suspense fallback={<BeatLoader color="#4277ff" css={override} size={10} />}>
          <Switch>
            {/* {process.env.BUILD === "EE" && <NavRoute exact path="/setting" component={Setting} />} */}
            {process.env.BUILD === "EE" && <NavRoute exact path="/member" component={Member} />}
            <NavRoute exact path="/" component={DashBoard} />
            <NavRoute exact path="/dashboard" component={DashBoard} />
            <NavRoute exact path="/dataset" component={DataSets} />
            <NavRoute exact path="/datasetAnalytics" component={DataSetAnalytics} />
            <NavRoute exact path="/label" component={LabelLayout} />
          </Switch>
        </Suspense>
      </div>

      <ReactTooltip place="bottom" delayShow={100} />
    </MainWrapper>
  )
}

export default Layouts
