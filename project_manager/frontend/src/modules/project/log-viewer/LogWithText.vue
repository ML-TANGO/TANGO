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
      :value="logText"
      background-color="#000"
      style="font-size: 12px"
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
      if (log?.message !== "\n") {
        if (log?.message?.trim()) {
          if (this.logText.length > 50000) {
            this.logText = this.logText.substring(10000);
          }

          this.logText += log.message;
        }
      }
      this.$nextTick(() => {
        const element = document.getElementById("log");
        if (element?.scrollHeight) element.scrollTop = element.scrollHeight;
      });
    }
  }
};
</script>
<style></style>
