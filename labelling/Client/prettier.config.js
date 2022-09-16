module.exports = {
  arrowParens: "avoid",
  htmlWhitespaceSensitivity: "css",
  printWidth: 140,
  semi: false,
  trailingComma: "none",
  tabWidth: 2,
  bracketSpacing: true,
  singleQuote: false,
  useTabs: false,
  rangeStart: 0,
  jsxBracketSameLine: false,
  overrides: [
    {
      files: "*.json",
      options: {
        printWidth: 200,
      },
    },
  ],
}
