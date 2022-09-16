var path = require("path")

var root = path.join(__dirname)

var config = {
  rootDir: root,
  // Targets ========================================================
  serveDir: path.join(root, ".serve"),
  distDir: path.join(root, "dist"),
  clientManifestFile: "manifest.webpack.json",
  clientStatsFile: "stats.webpack.json",

  // Source Directory ===============================================
  srcDir: path.join(root, "src"),
  srcServerDir: path.join(root, "src"),

  // HTML Layout ====================================================
  srcHtmlLayout: path.join(root, "public", "index.html"),
  srcFavicon: path.join(root, "public", "favicon.ico"),
  // Site Config ====================================================
  siteTitle: "BluAi",
  siteDescription: "BluAi-Vision AI",
  siteCannonicalUrl: "http://localhost:3001",
  siteKeywords: "",
  scssIncludes: []
}

module.exports = config
