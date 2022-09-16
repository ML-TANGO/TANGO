const path = require("path")
const webpack = require("webpack")
const HtmlWebpackPlugin = require("html-webpack-plugin")
const CircularDependencyPlugin = require("circular-dependency-plugin")
const ExtractCssChunks = require("extract-css-chunks-webpack-plugin")
const ManifestPlugin = require("webpack-manifest-plugin")
const TerserPlugin = require("terser-webpack-plugin")
const OptimizeCSSAssetsPlugin = require("optimize-css-assets-webpack-plugin")
const { CleanWebpackPlugin } = require("clean-webpack-plugin")
const svgToMiniDataURI = require("mini-svg-data-uri")
const config = require("./../config")
const crypto = require("crypto")

const isModuleCSS = module => {
  return module.type === "css/extract-css-chunks"
}
const BASE_PATH = process.env.BASE_PATH || "/"

module.exports = env => ({
  name: "BluAi",
  devtool: "cheap-module-source-map",
  target: "web",
  mode: "production",
  performance: {
    hints: false,
    maxEntrypointSize: 512000,
    maxAssetSize: 512000
  },
  node: {
    fs: "empty"
  },
  entry: {
    app: [path.join(config.srcDir, "index.js")]
  },
  output: {
    filename: "[name].[contenthash].bundle.js",
    chunkFilename: "[name].[contenthash].chunk.js",
    path: config.distDir,
    publicPath: BASE_PATH
  },
  resolve: {
    extensions: [".js", ".jsx", ".json", ".ts", ".tsx"],
    modules: ["node_modules", config.srcDir]
  },
  optimization: {
    minimizer: [
      new TerserPlugin({
        extractComments: false,
        cache: true,
        parallel: true,
        sourceMap: false, // Must be set to true if using source-maps in production
        terserOptions: {
          compress: {
            drop_console: true
          }
        }
      }),
      new OptimizeCSSAssetsPlugin()
    ],
    runtimeChunk: {
      //추가 부분
      name: "runtime"
    },
    splitChunks: {
      chunks: "all", // 모든 코드 분할
      minSize: 30000, // 최소 사이즈 30kb
      minChunks: 1,
      cacheGroups: {
        default: false,
        vendors: false,
        defaultVendors: false,
        framework: {
          chunks: "all",
          name: "framework",
          test: /(?<!node_modules.*)[\\/]node_modules[\\/](react|react-dom|react-router-dom)[\\/]/,
          priority: 40,
          enforce: true
        },
        lib: {
          test(module) {
            return module.size() > 80000 && /node_modules[/\\]/.test(module.identifier())
          },
          name(module) {
            const hash = crypto.createHash("sha1")

            if (isModuleCSS(module)) {
              module.updateHash(hash)

              return hash.digest("hex").substring(0, 8)
            } else {
              if (!module.libIdent) {
                throw new Error(`Encountered unknown module type: ${module.type}. Please open an issue.`)
              }
            }
            hash.update(module.libIdent({ context: __dirname }))

            return hash.digest("hex").substring(0, 8)
          },
          priority: 30,
          minChunks: 1,
          reuseExistingChunk: true
        },
        commons: {
          minChunks: 1, // entry points length
          priority: 20
        }
      }
    }
  },
  plugins: [
    new CircularDependencyPlugin({
      exclude: /a\.js|node_modules/,
      failOnError: true,
      allowAsyncCycles: false,
      cwd: process.cwd()
    }),
    new HtmlWebpackPlugin({
      template: config.srcHtmlLayout,
      favicon: config.srcFavicon,
      title: config.siteTitle,
      inject: true,
      chunksSortMode: "none"
    }),
    new webpack.HashedModuleIdsPlugin(),
    new webpack.DefinePlugin({
      "process.env.NODE_ENV": JSON.stringify("production"),
      "process.env.BASE_PATH": JSON.stringify(BASE_PATH),
      "process.env.BUILD": JSON.stringify(env.build)
    }),
    new ExtractCssChunks({
      filename: "[name].[contenthash].css",
      chunkFilename: "[id].[contenthash].css"
    }),
    new ManifestPlugin({
      fileName: config.clientManifestFile,
      publicPath: BASE_PATH
    }),
    new CleanWebpackPlugin({
      cleanAfterEveryBuildPatterns: ["dist"]
    })
  ],
  module: {
    rules: [
      {
        test: /\.js$/,
        include: config.srcDir,
        exclude: /node_modules/,
        use: "babel-loader"
      },
      //Modular Styles
      {
        test: /\.(sa|sc|c)ss$/,
        use: [
          { loader: "style-loader" },
          {
            loader: "css-loader",
            options: {
              importLoaders: 1,
              modules: true,
              localIdentName: "[path][name]__[local]--[hash:base64:5]",
              camelCase: true,
              sourceMap: true
            }
          },
          { loader: "postcss-loader" },
          {
            loader: "sass-loader",
            options: {
              sassOptions: {
                includePaths: config.scssIncludes
              }
            }
          }
        ],
        exclude: [path.resolve(config.srcDir, "Assets", "styles")],
        include: [config.srcDir]
      },
      // Project styles
      {
        test: /\.(sa|sc|c)ss$/,
        use: [
          ExtractCssChunks.loader,
          "css-loader",
          "postcss-loader",
          {
            loader: "sass-loader",
            options: {
              sassOptions: {
                includePaths: config.scssIncludes
              }
            }
          }
        ],
        include: [path.resolve(config.srcDir, "Assets", "styles")]
      },
      // Fonts
      {
        test: /\.(ttf|eot|woff|woff2|otf)$/i,
        loader: "file-loader",
        options: {
          name: "[name].[ext]",
          outputPath: "fonts",
          useRelativePaths: true
        }
      },
      // Files
      {
        test: /\.(jpe?g|png|gif|ico)$/i,
        use: [
          {
            loader: "url-loader",
            options: {
              name: "[name].[ext]",
              outputPath: "assets",
              useRelativePaths: true,
              esModule: false,
              limit: 10000 // 10kb 이하 파일은 url로 만들어 bundle 파일에 포함. 나머지는 파일
            }
          }
        ]
      },
      {
        test: /\.svg$/i,
        use: [
          {
            loader: "url-loader",
            options: {
              generator: content => svgToMiniDataURI(content.toString())
            }
          }
        ]
      }
    ]
  },
  devServer: {
    hot: false,
    contentBase: config.distDir,
    compress: true,
    historyApiFallback: {
      index: BASE_PATH
    },
    proxy: {
      "/api": "http://0.0.0.0:10236",
      "/qithum": "http://0.0.0.0:10236",
      "/static": "http://0.0.0.0:10236",
      "/socket.io": { target: "http://0.0.0.0:3000", ws: true }
    },
    host: "0.0.0.0",
    port: 10235
  }
})
