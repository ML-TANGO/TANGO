<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <div class="d-flex justify-space-between" style="width: 100%">
        <p class="text-h5 mb-4" style="color: #4a80ff">Hyperparameter</p>
      </div>
      <div style="width: 100%; height: 350px; backface-visibility: hidden; flex: 1 1 auto; overflow-y: auto">
        <!-- <v-textarea v-model="content" no-resize hide-details auto-grow></v-textarea> -->
        <div ref="editorContainer" style="width: 100%; height: 100%"></div>
      </div>
    </div>
    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="pre"> PREV </v-btn>
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="next"> CREATE </v-btn>
    </div>
  </div>
</template>
<script>
import { getProjectArgumentsFile, updateProjectArgumentsFile } from "@/api";

import * as monaco from "monaco-editor";

import { configureMonacoYaml } from "monaco-yaml";

export default {
  props: {
    project: {
      default: null
    }
  },

  data() {
    return {
      content: "",
      editor: null
    };
  },

  computed: {},

  async mounted() {
    try {
      const response = await getProjectArgumentsFile(this.project.id);
      console.log("response", response, response.content);

      this.content = response.content;

      configureMonacoYaml(monaco);
      this.editor = monaco.editor.create(this.$refs.editorContainer, {
        value: this.content, // 초기 값
        language: "yaml"
      });
    } catch {
      this.items = [];
    }
  },

  methods: {
    pre() {
      this.$emit("prev");
    },

    async next() {
      updateProjectArgumentsFile(this.project.id, this.editor.getValue());
      this.$emit("create");
    }
  }
};
</script>
<style lang="scss" scoped></style>
