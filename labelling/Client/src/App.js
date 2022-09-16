import React, { useEffect, useState } from "react"
import { Provider } from "react-redux"
import { BrowserRouter } from "react-router-dom"
import { hot } from "react-hot-loader"
import { CookiesProvider } from "react-cookie"
import { ToastContainer } from "react-toastify"

import * as AuthApi from "../src/Config/Services/AuthApi"

import configureStore from "./configureStore"
import Router from "./Router/Router"

import "./Assets/styles/modular/app.scss"

const store = configureStore()

const App = () => {
  const [isLoaded, setIsLoaded] = useState(false)
  const [result, setReulst] = useState({})
  useEffect(() => {
    if (process.env.NODE_ENV === "development") {
      setIsLoaded(true)
      setReulst({ status: 1, msg: "development mode!" })
    } else {
      AuthApi._getAuthCheck({ BUILD: process.env.BUILD })
        .then(result => {
          setIsLoaded(true)
          setReulst(result)
        })
        .catch(e => {
          setIsLoaded(true)
          setReulst({ status: 0 })
          console.log(e)
        })
    }
  }, [])

  return (
    <CookiesProvider>
      <Provider store={store}>
        <BrowserRouter basename="/">{isLoaded ? <Router result={result} /> : null}</BrowserRouter>
        <ToastContainer
          position="top-right"
          autoClose={2000}
          hideProgressBar
          newestOnTop
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
        />
      </Provider>
    </CookiesProvider>
  )
}

export default hot(module)(App)
