module.exports = api => {
  const babelEnv = api.env()
  const babelVer = api.version
  console.log(` -- Babel(${babelVer}) Mode : ${babelEnv} --`)
  
  const presets = [
    [
      "@babel/preset-env",
      {
        modules: false,
        // 최신 문법 지원을 위한 targets 추가
        targets: {
          browsers: ["last 2 versions", "not dead"]
        }
      }
    ],
    "@babel/preset-react"
  ]
  
  const plugins = [
    "@babel/plugin-proposal-class-properties",
    "@babel/plugin-syntax-dynamic-import",
    // Optional Chaining 지원 추가
    "@babel/plugin-proposal-optional-chaining",
    // Nullish Coalescing 지원 추가
    "@babel/plugin-proposal-nullish-coalescing-operator",
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

