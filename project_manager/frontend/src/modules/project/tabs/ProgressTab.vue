<template>
  <div class="d-flex flex-column" style="gap: 15px; margin-left: -16px">
    <div>
      <div style="width: 100%" class="d-flex justify-center">
        <v-btn class="mb-3" color="#4a80ff" width="380" dark @click="autoCreate">신경망 자동 생성</v-btn>
      </div>
      <h4 class="ml-3 mb-3">Progress - Auto NN</h4>
      <v-card color="#DFDFDF" class="" style="border-radius: 4px" height="180">
        <div>
          <ProgressCanvas :running="projectInfo?.container" @start="start" />

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
import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProgressCanvas from "@/modules/project/components/ProgressCanvas.vue";

import { startContainer, stopContainer, getStatusResult } from "@/api";
export default {
  components: { ProgressCanvas },
  props: {
    projectInfo: {
      default: () => ({})
    }
  },
  data() {
    return {
      vale: ``,
      running: ""
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"])
  },

  mounted() {
    this.$nextTick(() => {
      const element = document.getElementById("log");
      element.scrollTop = element.scrollHeight;
    });

    this.$EventBus.$on("logUpdate", this.updateLog);
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT
    }),

    updateLog(log) {
      console.log("updateLog", log);
      if (log.message !== "\n") {
        this.vale += log.message;
      }
      this.$nextTick(() => {
        const element = document.getElementById("log");
        element.scrollTop = element.scrollHeight;
      });
    },

    start(container) {
      console.log("projectInfo", this.projectInfo);

      this.$swal
        .fire({
          title: `${container}를 실행하시겠습니까?`,
          text: "",
          icon: "warning",
          showCancelButton: true,
          confirmButtonColor: "#3085d6",
          cancelButtonColor: "#d33",
          confirmButtonText: "확인",
          cancelButtonText: "취소"
        })
        .then(async result => {
          if (result.isConfirmed) {
            startContainer(container, this.projectInfo.create_user, this.projectInfo.id);
          }
        });

      this.SET_PROJECT({
        container: container
      });
    },

    async autoCreate() {
      // 실행중인 컨테이너가 있다면 종료
      if (this.projectInfo?.container && this.projectInfo?.container !== "" && this.projectInfo?.container !== "init") {
        this.$swal
          .fire({
            title: `실행 중인 작업이 있습니다.`,
            text: "다시 시작 하시겠습니까?",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#3085d6",
            cancelButtonColor: "#d33",
            confirmButtonText: "확인",
            cancelButtonText: "취소"
          })
          .then(async result => {
            if (result.isConfirmed) {
              await getStatusResult(this.projectInfo.id);
              await stopContainer(this.projectInfo.container, this.projectInfo.create_user, this.projectInfo.id);
              startContainer("bms", this.projectInfo.create_user, this.projectInfo.id);
            }
          });
      } else {
        await getStatusResult(this.projectInfo.id);
        startContainer("bms", this.projectInfo.create_user, this.projectInfo.id);
      }
    }
  }
};
</script>
<style lang="scss" scoped>
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
