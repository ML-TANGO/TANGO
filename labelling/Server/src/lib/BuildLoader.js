class BuildLoader {
  constructor() {
    this.BUILD = "CE"
  }
  setBUILD(build) {
    this.BUILD = build
  }

  getBUILD() {
    return this.BUILD
  }
}

module.exports = new BuildLoader()
