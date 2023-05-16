const { defineConfig } = require("@vue/cli-service");
module.exports = defineConfig({
  transpileDependencies: ["vuetify"],
  outputDir: "build",
  publicPath: "/",
  assetsDir: "static",
  devServer: {
    allowedHosts: "all"
  }
});
