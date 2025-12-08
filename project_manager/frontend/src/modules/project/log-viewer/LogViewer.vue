<template>
  <div class="log-viewer-shell">
    <LogWithText v-if="viewerMode === ViewerMode.TEXT" />
    <AutonnChatViewer v-else-if="viewerMode === ViewerMode.CHAT" />
    <AutonnLogViewer v-else :data="autonn_status"></AutonnLogViewer>
    <!-- <ModelViewer v-else-if="viewerMode === ViewerMode.MODEL_VIEW" />
    <LogWithChart v-else :data="autonn_status" /> -->
  </div>
</template>
<script>
import { mapState } from "vuex";
import { ProjectNamespace } from "@/store/modules/project";

import LogWithText from "./LogWithText.vue";
import AutonnLogViewer from "./AutonnLogViewer.vue";
import AutonnChatViewer from "../model-viewer/AutonnChatViewer.vue";

import { TaskType, ContainerName, ViewerMode, ProjectStatus } from "@/shared/enums";

export default {
  components: { LogWithText, AutonnLogViewer, AutonnChatViewer },

  data() {
    return {
      ViewerMode
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project", "autonn_status"]),

    viewerMode() {
      if (
        this.project.container_status === ProjectStatus.STARTED ||
        this.project.container_status === ProjectStatus.STOPPED ||
        this.project.container_status === ProjectStatus.COMPLETED
      ) {
        return ViewerMode.TEXT;
      }
      if (this.project.container === ContainerName.AUTO_NN) {
        if (this.project.task_type === TaskType.CHAT) {
          return ViewerMode.CHAT;
        } else if (this.autonn_status?.progress >= 1 && this.autonn_status?.progress <= 1) {
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
<style>
.log-viewer-shell {
  width: 100%;
  min-height: 0;
}
</style>
