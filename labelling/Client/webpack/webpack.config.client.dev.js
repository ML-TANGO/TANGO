const path = require("path")
const webpack = require("webpack")
const HtmlWebpackPlugin = require("html-webpack-plugin")
const CircularDependencyPlugin = require("circular-dependency-plugin")
const ManifestPlugin = require("webpack-manifest-plugin")
const { CleanWebpackPlugin } = require("clean-webpack-plugin")
const ESLintPlugin = require("eslint-webpack-plugin")
const svgToMiniDataURI = require("mini-svg-data-uri")
// const BundleAnalyzerPlugin = require("webpack-bundle-analyzer").BundleAnalyzerPlugin
const config = require("../config")
const crypto = require("crypto")

const isModuleCSS = module => {
  return module.type === "css/extract-css-chunks"
}

const BASE_PATH = process.env.BASE_PATH || "/"

module.exports = env => {
  return {
    name: "BluAi_Dev",
    devtool: "inline-source-map",
    target: "web",
    mode: "development",
    node: {
      fs: "empty"
    },
    entry: {
      app: [path.join(config.srcDir, "index.js")]
    },
    output: {
      filename: "[name].[hash].bundle.js",
      chunkFilename: "[name].[hash].chunk.js",
      path: config.distDir,
      publicPath: BASE_PATH
    },
    resolve: {
      extensions: [".js", ".jsx", ".json", ".ts", ".tsx"],
      alias: {
        "react-dom": "@hot-loader/react-dom"
      },
      modules: ["node_modules", config.srcDir]
    },
    optimization: {
      runtimeChunk: {
        name: "runtime" // 추가 부분
      },
      splitChunks: {
        chunks: "all", // 모든 코드 분할
        minSize: 30000, // 최소 사이즈 30kb
        minChunks: 1,
        cacheGroups: {
          default: {
            minChunks: 2,
            priority: -20,
            reuseExistingChunk: true
          },
          defaultVendors: {
            test: /[\\/]node_modules[\\/]/,
            priority: -10
          },
          reactBundle: {
            test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
            name: "react.bundle",
            priority: 2,
            minSize: 100
          }
          // framework: {
          //   chunks: "all",
          //   name: "framework",
          //   test: /(?<!node_modules.*)[\\/]node_modules[\\/](react|react-dom|react-router-dom)[\\/]/,
          //   priority: 40,
          //   enforce: true
          // },
          // lib: {
          //   test(module) {
          //     return module.size() > 80000 && /node_modules[/\\]/.test(module.identifier())
          //   },
          //   name(module) {
          //     const hash = crypto.createHash("sha1")

          //     if (isModuleCSS(module)) {
          //       module.updateHash(hash)

          //       return hash.digest("hex").substring(0, 8)
          //     } else {
          //       if (!module.libIdent) {
          //         throw new Error(`Encountered unknown module type: ${module.type}. Please open an issue.`)
          //       }
          //     }
          //     hash.update(module.libIdent({ context: __dirname }))

          //     return hash.digest("hex").substring(0, 8)
          //   },
          //   priority: 30,
          //   minChunks: 1,
          //   reuseExistingChunk: true
          // },
          // commons: {
          //   minChunks: 1, // entry points length
          //   priority: 20
          // }
        }
      }
    },
    // stats: {
    //   // Examine all modules
    //   maxModules: Infinity,
    //   // Display bailout reasons
    //   optimizationBailout: true
    // },
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
      new webpack.DefinePlugin({
        "process.env.NODE_ENV": JSON.stringify("development"),
        "process.env.BASE_PATH": JSON.stringify(BASE_PATH),
        "process.env.BUILD": JSON.stringify(env.build)
      }),
      new webpack.HotModuleReplacementPlugin({ multiStep: false }),
      new ManifestPlugin({
        fileName: config.clientManifestFile,
        publicPath: BASE_PATH
      }),
      new CleanWebpackPlugin({
        cleanAfterEveryBuildPatterns: ["dist"]
      }),
      new ESLintPlugin({ files: "src/**/*.js" })
      // new BundleAnalyzerPlugin({
      //   analyzerHost: "127.0.0.1",
      //   analyzerPort: 8080
      // })
    ],
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          use: "babel-loader"
        },
        // Modular Styles
        {
          test: /\.(sa|sc|c)ss$/,
          use: [
            { loader: "style-loader" },
            {
              loader: "css-loader",
              options: {
                importLoaders: 1,
                modules: true,
                localIdentName: "[path][name]",
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
        // Project Styles
        {
          test: /\.(sa|sc|c)ss$/,
          use: [
            "style-loader",
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
      hot: true,
      inline: true,
      contentBase: config.serveDir,
      compress: true,
      historyApiFallback: {
        index: BASE_PATH
      },
      proxy: {
        "/api": { target: "http://localhost:10236", proxyTimeout: 5 * 60 * 1000, timeout: 5 * 60 * 1000 },
        "/qithum": "http://localhost:10236",
        "/static": "http://localhost:10236",
        "/socket.io": { target: "http://localhost:3000", ws: true }
      },
      // proxy: {
      //   "/api": "http://10.140.141.202:9140",
      //   "/qithum": "http://10.140.141.202:9140",
      //   "/static": "http://10.140.141.202:9140",
      //   "/socket.io": { target: "http://10.140.141.202:9140", ws: true }
      // },
      // proxy: {
      //   "/api": "http://192.168.0.4:10236",
      //   "/qithum": "http://192.168.0.4:10236",
      //   "/static": "http://192.168.0.4:10236",
      //   "/socket.io": { target: "http://192.168.0.4:3000", ws: true }
      // },
      host: "0.0.0.0",
      open: true,
      port: 10235
    }
  }
}
