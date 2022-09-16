module.exports = {
  extends: ["eslint:recommended", "plugin:react/recommended", "plugin:react-hooks/recommended"],
  plugins: ["react", "react-hooks"],
  parser: "@babel/eslint-parser",
  parserOptions: {
    ecmaVersion: 2018,
    sourceType: "module",
    ecmaFeatures: {
      jsx: true,
      experimentalObjectRestSpread: true
    }
  },
  settings: {
    react: {
      version: "detect"
    }
  },
  env: {
    es6: true,
    browser: true,
    node: true
  },
  rules: {
    "no-console": "off",
    quotes: "off",
    "arrow-body-style": "off",
    "react/prop-types": "off",
    "no-case-declarations": "off",
    "react/display-name": "off",
    "no-unused-vars": "warn",
    "react-hooks/rules-of-hooks": "error", // Checks rules of Hooks
    "react-hooks/exhaustive-deps": "off"
  }
}
