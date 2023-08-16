<template>
  <div class="d-flex flex-column" style="gap: 15px; margin-left: -16px">
    <div>
      <div style="width: 100%" class="d-flex justify-center">
        <div style="gap: 15px" class="d-flex align-center">
          <v-btn class="mb-3" color="#4a80ff" width="380" dark @click="menualCreate">
            Manually Generation of Neural Networks
          </v-btn>

          <v-btn class="mb-3" color="#4a80ff" width="380" dark @click="autoCreate">
            Automatic Generation of Neural Networks
          </v-btn>
        </div>
      </div>
      <h4 class="ml-3 mb-3">
        Progress - {{ showContainerName(projectInfo?.container) }} {{ projectInfo?.container_status }}
      </h4>
      <v-card color="#DFDFDF" class="" style="border-radius: 4px" height="180">
        <div style="height: 100%">
          <ProgressCanvas
            :running="projectInfo?.container"
            :status="projectInfo?.container_status"
            :userEdit="project.deploy_user_edit === 'yes'"
            @start="start"
            @immediateLaunch="immediateLaunch"
          />
        </div>
      </v-card>
    </div>
    <div>
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
      <v-card color="#000" class="ml-3" style="border-radius: 4px">
        <v-textarea
          ref="logs"
          id="log"
          class="mb-5"
          dark
          filled
          :value="vale"
          style="font-size: 12px"
          readonly
          hide-details
          autofocus
        ></v-textarea>
      </v-card>
    </div>
  </div>
</template>
<script>
import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProgressCanvas from "@/modules/project/components/ProgressCanvas.vue";

import { ProjectType } from "@/shared/consts";

import { containerStart, updateProjectType } from "@/api";
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
      running: "",
      copyFailed: false,
      copySuccess: false
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
      if (log.message !== "\n") {
        if (log.message.trim()) {
          this.vale += log.message;
        }
      }
      this.$nextTick(() => {
        const element = document.getElementById("log");
        element.scrollTop = element.scrollHeight;
      });
    },

    start(container) {
      const containerName =
        container === "bms"
          ? "BMS"
          : container === "yoloe"
          ? "Auto NN"
          : container === "codeGen"
          ? "Code Gen"
          : "Image Deploy";

      this.$swal
        .fire({
          title: `${containerName}를 실행하시겠습니까?`,
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
            this.$emit("restart", container);
            await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
            await this.containerStartRequest(container);
            this.SET_PROJECT({ project_type: ProjectType.MANUAL });
          }
        });

      this.SET_PROJECT({
        container: container
      });
    },

    async immediateLaunch(container) {
      console.log("immediateLaunch", container);
      this.$emit("restart", container);
      await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
      await this.containerStartRequest(container);
      this.SET_PROJECT({ project_type: ProjectType.MANUAL, container: container });
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
              await updateProjectType(this.projectInfo.id, ProjectType.AUTO);
              await this.containerStartRequest("bms");
            }
          });
      } else {
        await updateProjectType(this.projectInfo.id, ProjectType.AUTO);
        await this.containerStartRequest("bms");
      }

      this.SET_PROJECT({
        project_type: ProjectType.AUTO
      });

      this.$emit("start");
    },

    async menualCreate() {
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
              await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
            }
          });
      } else {
        await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
      }

      this.SET_PROJECT({
        project_type: ProjectType.MANUAL
      });
    },

    showContainerName(container) {
      if (container) {
        if (container.toLowerCase() === "bms") {
          return "BMS";
        } else if (container.toLowerCase() === "yoloe") {
          return "Auto NN";
        } else if (container.toLowerCase() === "codegen") {
          return "Code Gen";
        } else if (container.toLowerCase() === "imagedeploy") {
          return "Image Deploy";
        } else {
          return "";
        }
      } else {
        return "";
      }
    },

    clipboardCopy() {
      this.$copyText(this.vale).then(
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

    async containerStartRequest(container) {
      const res = await containerStart(container, this.projectInfo.create_user, this.projectInfo.id);
      this.vale += res.message;
      this.vale += res.response;
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
