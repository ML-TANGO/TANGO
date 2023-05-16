<template>
  <div class="d-flex flex-column" style="gap: 15px; margin-left: -16px">
    <div>
      <div style="width: 100%" class="d-flex justify-center">
        <v-btn class="mb-3" color="#4a80ff" width="380" dark>신경망 자동 생성</v-btn>
      </div>
      <h4 class="ml-3 mb-3">Progress - Auto NN</h4>
      <v-card color="#DFDFDF" class="" style="border-radius: 4px" height="180">
        <!-- <v-card color="#DFDFDF" class="ml-3 px-10 pb-5" style="border-radius: 4px" height="180"> -->
        <!-- <div style="gap: 35px; font-size: 12px; height: 100%" class="d-flex align-center justify-space-around">
          <v-btn
            style="border-radius: 4px; width: 120px; height: 40px; background: #4a80ff; color: white; border: none"
            class="d-flex align-center justify-center end btn"
            outlined
            @click="adasd = 1"
          >
            BMS
          </v-btn>
          <v-divider style="border-width: 2px; border-radius: 1px" />
          <v-btn
            style="border-radius: 4px; width: 120px; height: 40px; background: #4a80ff; color: white; border: none"
            outlined
            class="d-flex align-center justify-center box btn"
          >
            Auto NN
          </v-btn>
          <v-divider style="border-width: 2px; border-radius: 1px" />
          <v-btn
            style="border-radius: 4px; width: 120px; height: 40px; background: #4a80ff; color: white; border: none"
            outlined
            class="d-flex align-center justify-center btn"
          >
            Image GEN
          </v-btn>
          <v-divider style="border-width: 2px; border-radius: 1px" />
          <v-btn
            style="border-radius: 4px; width: 120px; height: 40px; background: #4a80ff; color: white; border: none"
            outlined
            class="d-flex align-center justify-center btn"
          >
            Image Deploy
          </v-btn>
          <v-divider style="border-width: 2px; border-radius: 1px" />
          <v-btn
            style="border-radius: 4px; width: 120px; height: 40px; background: #4a80ff; color: white; border: none"
            class="d-flex align-center justify-center btn"
          >
            Run Image
          </v-btn>
        </div> -->
        <div>
          <ProgressCanvas :running="running" @start="start" />

          <div style="width: 268px; height: 35px; position: relative; top: -64px; left: 250px" class="visual">
            <v-btn
              style="
                position: absolute;
                bottom: -20px;
                left: calc(50% - 60px);
                border-radius: 4px;
                width: 120px;
                height: 40px;
                background: #4a80ff;
                color: white;
                border: none;
                padding-bottom: -20px;
              "
              class="d-flex align-center justify-center"
            >
              Visualization
            </v-btn>
          </div>
        </div>
      </v-card>
    </div>
    <div>
      <h4 class="ml-3 mb-3">Log</h4>
      <v-card color="#000" class="ml-3" style="border-radius: 4px" height="200">
        <v-textarea
          ref="logs"
          id="log"
          dark
          filled
          no-resize
          :value="vale"
          height="200"
          style="font-size: 12px"
          readonly
        ></v-textarea>
      </v-card>
    </div>
  </div>
</template>
<script>
import ProgressCanvas from "@/modules/project/components/ProgressCanvas.vue";

import { startContainer } from "@/api";
export default {
  components: { ProgressCanvas },
  props: {
    projectInfo: () => ({})
  },
  data() {
    return {
      vale: `2023-04-27 14:07:19        You must specify POSTGRES_PASSWORD to a non-empty value for the`,
      running: ""
    };
  },

  mounted() {
    this.$nextTick(() => {
      const element = document.getElementById("log");
      element.scrollTop = element.scrollHeight;
    });
  },

  methods: {
    updateLog() {
      this.$nextTick(() => {
        const element = document.getElementById("log");
        element.scrollTop = element.scrollHeight;
      });
    },

    start(container) {
      startContainer(container, this.projectInfo.create_user, this.projectInfo.id);
    }
  }
};
</script>
<style lang="scss" scoped>
.box {
  --border-size: 2px;
  --border-angle: 0turn;

  background-image: conic-gradient(from var(--border-angle), #ff3d54, #ff3d54 50%, #ff3d54),
    conic-gradient(from var(--border-angle), transparent 0%, #ff3d54, #ff3d54) !important;
  background-size: calc(100% - (var(--border-size) * 2)) calc(100% - (var(--border-size) * 2)), cover !important;
  background-position: center center !important;
  background-repeat: no-repeat !important;
  background-color: rgba(255, 61, 84, 0.2) !important;

  color: black !important;

  animation: bg-spin 2s linear infinite;
  @keyframes bg-spin {
    to {
      --border-angle: 1turn;
    }
  }
}

@property --border-angle {
  syntax: "<angle>";
  inherits: true;
  initial-value: 0turn;
}

.end {
  background-color: #cfd2cf !important;
  color: black !important;
}

.visual {
  border-left: 3px dashed rgba(0, 0, 0, 0.12);
  border-right: 3px dashed rgba(0, 0, 0, 0.12);
  border-bottom: 3px dashed rgba(0, 0, 0, 0.12);
  z-index: 0;
}

.btn {
  z-index: 15;
}
</style>
