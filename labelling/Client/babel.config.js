module.exports = api => {
  const babelEnv = api.env()
  const babelVer = api.version
  console.log(` -- Babel(${babelVer}) Mode : ${babelEnv} --`)
  const presets = [
    [
      "@babel/preset-env",
      {
        modules: false
      }
    ],
    "@babel/preset-react"
  ]
  const plugins = [
    "@babel/plugin-proposal-class-properties",
    "@babel/plugin-syntax-dynamic-import",
    "universal-import",
    "react-hot-loader/babel",
    "emotion"
  ]
  const env = {
    development: {
      compact: false,
      plugins: ["react-hot-loader/babel"]
    }
  }
  return {
    presets,
    plugins,
    env
  }
}
