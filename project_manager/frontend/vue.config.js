const { defineConfig } = require("@vue/cli-service");
const MonacoWebpackPlugin = require("monaco-editor-webpack-plugin");

module.exports = defineConfig({
  transpileDependencies: ["vuetify"],
  outputDir: "build",
  publicPath: "/",
  assetsDir: "static",
  devServer: {
    allowedHosts: "all",
    client: {
      overlay: false
    }
  },
  configureWebpack: {
    plugins: [new MonacoWebpackPlugin()]
  }
});
