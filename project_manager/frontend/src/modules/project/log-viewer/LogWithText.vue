<template>
  <div style="width: 100%; height: 100%">
    <div class="d-flex justify-space-between align-center" style="width: 100%">
      <h4 class="ml-3 mb-3">Log</h4>

      <v-tooltip left>
        <template v-slot:activator="{ on, attrs }">
          <v-btn icon v-bind="attrs" v-on="on">
            <v-icon v-if="copySuccess">mdi-clipboard-check-multiple</v-icon>
            <v-icon v-else-if="copyFailed">mdi-clipboard-off</v-icon>
            <v-icon v-else @click="clipboardCopy">mdi-clipboard-text</v-icon>
          </v-btn>
        </template>
        <span v-if="copySuccess" style="font-size: 10px">Copied</span>
        <span v-else-if="copyFailed" style="font-size: 10px">Can not copy</span>
        <span v-else style="font-size: 10px">Copy Log</span>
      </v-tooltip>
    </div>
    <v-textarea
      ref="logs"
      id="log"
      class="mb-5 ma-1 log-area"
      dark
      filled
      rows="30"
      :value="logText"
      background-color="#000"
      style="font-size: 16px; line-height: 1.5; min-height: 60vh; max-height: 100vh"
      readonly
      hide-details
    ></v-textarea>
  </div>
</template>
<script>
export default {
  data() {
    return {
      copyFailed: false,
      copySuccess: false,
      logText: ""
    };
  },

  mounted() {
    this.$nextTick(() => {
      const element = document.getElementById("log");
      if (element?.scrollHeight) element.scrollTop = element.scrollHeight;
    });

    this.$EventBus.$off("logUpdate");
    this.$EventBus.$on("logUpdate", this.updateLog);
  },

  methods: {
    clipboardCopy() {
      this.$copyText(this.logText).then(
        () => {
          this.copySuccess = true;
          setTimeout(() => {
            this.copySuccess = false;
          }, 1000);
        },
        () => {
          this.copyFailed = true;
          setTimeout(() => {
            this.copyFailed = false;
          }, 1000);
        }
      );
    },

    updateLog(log) {
      const message = this.extractDockerLog(log);
      if (!message) return;

      if (this.logText.length > 50000) {
        this.logText = this.logText.substring(10000);
      }

      this.logText += message.endsWith("\n") ? message : `${message}\n`;

      this.$nextTick(() => {
        const element = document.getElementById("log");
        if (element?.scrollHeight) element.scrollTop = element.scrollHeight;
      });
    },

    extractDockerLog(log) {
      const raw = typeof log === "string" ? log : log?.message;
      if (!raw || raw === "\n") return "";

      const lines = raw.split(/\r?\n/).filter(Boolean);

      // For autonn logs, they may already be prefiltered; don't drop anything
      return lines.join("\n");
    }
  }
};
</script>
<style>
.log-area textarea {
  font-family: "JetBrains Mono", "Fira Code", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-variant-ligatures: none;
  font-size: 16px !important;
  line-height: 1.6 !important;
  min-height: 60vh !important;
  max-height: 100vh !important;
}
.log-area .v-textarea__slot textarea,
.log-area .v-input__slot textarea {
  font-family: "JetBrains Mono", "Fira Code", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-variant-ligatures: none;
  font-size: 16px !important;
  line-height: 1.6 !important;
  min-height: 60vh !important;
  max-height: 100vh !important;
}
</style>
