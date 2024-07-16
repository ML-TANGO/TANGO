<template>
  <div style="width: 100%; height: 100%">
    <LogWithText v-if="viewerMode === ViewerMode.TEXT" />
    <ModelViewer v-else-if="viewerMode === ViewerMode.MODEL_VIEW" />
    <LogWithChart v-else :data="autonn_status" />
  </div>
</template>
<script>
import { mapState } from "vuex";
import { ProjectNamespace } from "@/store/modules/project";

import LogWithText from "./LogWithText.vue";
import LogWithChart from "./LogWithChart.vue";
import ModelViewer from "@/modules/project/model-viewer/ModelViewer.vue";
import { ContainerName, ViewerMode } from "@/shared/enums";

export default {
  components: { LogWithText, LogWithChart, ModelViewer },

  data() {
    return {
      ViewerMode
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project", "autonn_status"]),

    viewerMode() {
      if (this.project.container_status === "started") {
        return ViewerMode.TEXT;
      }
      if (this.project.container === ContainerName.AUTO_NN) {
        if (this.autonn_status?.progress >= 1 && this.autonn_status?.progress <= 1) {
          return ViewerMode.MODEL_VIEW;
          // return ViewerMode.CHART;
        } else {
          return ViewerMode.CHART;
        }
      }
      return ViewerMode.TEXT;
    }
  }
};
</script>
<style lang=""></style>
