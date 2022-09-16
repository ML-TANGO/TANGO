import React, { useEffect, useState } from "react"
import { Route, Switch } from "react-router-dom"
import { useHistory } from "react-router-dom"
import BeatLoader from "react-spinners/BeatLoader"
import { css } from "@emotion/core"

// import LogInLayout from "Layouts/LogInLayout"
// import Layout from "Layouts/Layouts"
import RegisterLicense from "../Layouts/Contents/License/RegisterLicense"
import ApplyLicense from "../Layouts/Contents/License/ApplyLicense"
import DownloadPretrain from "../Layouts/Contents/License/DownloadPretrain"

const LogInLayout = React.lazy(() => import("Layouts/LogInLayout"))
const Layout = React.lazy(() => import("Layouts/Layouts"))
// const RegisterLicense = React.lazy(() => import("../Layouts/Contents/License/RegisterLicense"))
// const ApplyLicense = React.lazy(() => import("../Layouts/Contents/License/ApplyLicense"))
// const DownloadPretrain = React.lazy(() => import("../Layouts/Contents/License/DownloadPretrain"))
// const LogIn = React.lazy(() => import("Layouts/LogInLayout"))
// const Layout = React.lazy(() => import("Layouts/Layouts"))

const override = css`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
`

const LogInLayoutComponent = () => {
  return (
    <React.Suspense fallback={<BeatLoader css={override} size={10} color={"#1777cb"} />}>
      <LogInLayout></LogInLayout>
    </React.Suspense>
  )
}

const LayoutComponent = () => {
  return (
    <React.Suspense fallback={<BeatLoader css={override} size={10} color={"#1777cb"} />}>
      <Layout></Layout>
    </React.Suspense>
  )
}

const Router = ({ result }) => {
  const history = useHistory()
  const [isLicense, setIsLicense] = useState(false)

  useEffect(() => {
    if (history.location?.state?.result) {
      if (history.location?.state?.result.status === 1) setIsLicense(true)
      delete history.location?.state
      history.replace()
    }
  }, [history.location?.state?.result])

  return (
    <Switch>
      {result.status === 1 || isLicense ? (
        <Switch>
          {/* license 등록 성공 이후 Router */}
          {/* <Route exact path="/" component={LogInLayoutComponent} /> */}
          <Route exact path="/downloadPretrain" component={DownloadPretrain} />
          <Route path="/" component={LayoutComponent} />
        </Switch>
      ) : (
        <Switch>
          {/* license 등록  Router */}
          <Route exact path="/registerLicense" component={RegisterLicense} />
          <Route exact path="/applyLicense" component={ApplyLicense} />
          <Route exact path="/downloadPretrain" component={DownloadPretrain} />
          {/* <Route exact path="/login" component={LogInLayout} /> */}
          <Route path="/" component={RegisterLicense} />
        </Switch>
      )}
    </Switch>
  )
}

export default Router
