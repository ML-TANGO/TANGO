<template>
  <div class="d-flex flex-column" style="gap: 15px; margin-left: -16px">
    <div>
      <div style="width: 100%" class="d-flex justify-center">
        <div style="gap: 15px" class="d-flex align-center">
          <v-btn
            class="mb-3"
            color="#4a80ff"
            width="380"
            dark
            @click="menualCreate"
            :disabled="projectInfo?.container_status === 'started' || projectInfo?.container_status === 'running'"
          >
            Manually Generation of Neural Networks
          </v-btn>

          <v-btn
            class="mb-3"
            color="#4a80ff"
            width="380"
            dark
            @click="autoCreate"
            :disabled="projectInfo?.container_status === 'started' || projectInfo?.container_status === 'running'"
          >
            Automatic Generation of Neural Networks
          </v-btn>
        </div>
      </div>
      <h4 class="ml-3 mb-3">
        Progress - {{ DisplayName[projectInfo?.container] }} {{ projectInfo?.container_status }}
      </h4>
      <v-card color="#DFDFDF" class="ma-1" style="border-radius: 4px" height="180">
        <div style="height: 100%">
          <ProgressCanvas
            :running="projectInfo?.container"
            :status="projectInfo?.container_status"
            :userEdit="project.deploy_user_edit === 'yes'"
            :workflow="project?.workflow"
            @start="start"
            @showVis2code="showVis2code"
            @immediateLaunch="immediateLaunch"
          />
        </div>
      </v-card>
      <v-btn
        @click="open = !open"
        v-if="project.task_type === TaskType.CLASSIFICATION && projectInfo?.container !== ContainerName.BMS"
        style="background-color: #fff"
        class="ml-1 elevation-0"
      >
        <div v-if="!open">
          <v-icon>mdi-chevron-down</v-icon>

          Open VISUALIZATION
        </div>

        <div v-else>
          <v-icon>mdi-chevron-up</v-icon>
          Close VISUALIZATION
        </div>
      </v-btn>
      <v-banner
        v-model="open"
        style="padding: 0px !important"
        class="custom"
        v-if="project.task_type === TaskType.CLASSIFICATION"
      >
        <v-card style="height: 1080px; overflow: none" class="mt-5">
          <iframe :src="HongIKVis2Code" title="내용" width="100%" height="100%"></iframe>
        </v-card>
      </v-banner>
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
      <v-textarea
        ref="logs"
        id="log"
        class="mb-5 ma-1 log-area"
        dark
        filled
        :value="vale"
        background-color="#000"
        style="font-size: 12px"
        readonly
        hide-details
      ></v-textarea>
    </div>
  </div>
</template>
<script>
import Swal from "sweetalert2";
import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProgressCanvas from "@/modules/project/components/ProgressCanvas.vue";

import { ProjectType } from "@/shared/consts";
import { DisplayName, TaskType, ContainerName } from "@/shared/enums";

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
      copySuccess: false,
      isVis2Code: false,
      open: false,
      DisplayName,
      TaskType,
      ContainerName
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"]),

    HongIKVis2Code() {
      if (process.env.NODE_ENV === "production") {
        const host = window.location.hostname;
        return `http://${host}:8091`;
      } else {
        return `${process.env.VUE_APP_ROOT_HOST}:8091`;
      }
    }
  },

  mounted() {
    this.$nextTick(() => {
      const element = document.getElementById("log");
      element.scrollTop = element.scrollHeight;
    });

    this.$EventBus.$on("logUpdate", this.updateLog);
    this.$EventBus.$on("control_Vis2Code", status => {
      this.isVis2Code = status;
      this.open = status;
    });
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT
    }),

    showVis2code() {
      this.isVis2Code = true;
      this.open = true;
    },

    updateLog(log) {
      if (log?.message !== "\n") {
        if (log?.message?.trim()) {
          if (this.vale.length > 50000) {
            this.vale = this.vale.substring(10000);
          }

          this.vale += log.message;
        }
      }
      this.$nextTick(() => {
        const element = document.getElementById("log");
        element.scrollTop = element.scrollHeight;
      });
    },

    start(container) {
      const containerName = DisplayName[container];

      if (this.project.container_status === "running" || this.project.container_status === "started") {
        Swal.fire({
          title: "이미 컨테이너가 실행 중입니다.",
          icon: "error",
          text: ""
        });
        return;
      }

      Swal.fire({
        title: `${containerName}를 실행하시겠습니까?`,
        text: "",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "확인",
        cancelButtonText: "취소"
      }).then(async result => {
        if (result.isConfirmed) {
          this.SET_PROJECT({
            container: container,
            container_status: "started"
          });

          this.$emit("restart", container);
          await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
          await this.containerStartRequest(container);
          this.SET_PROJECT({ project_type: ProjectType.MANUAL });
        }
      });
    },

    async immediateLaunch(container) {
      this.$emit("restart", container);
      await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
      await this.containerStartRequest(container);
      this.SET_PROJECT({ project_type: ProjectType.MANUAL, container: container });
    },

    async autoCreate() {
      // 실행중인 컨테이너가 있다면 종료
      if (this.projectInfo?.container && this.projectInfo?.container !== "" && this.projectInfo?.container !== "init") {
        Swal.fire({
          title: `실행 중인 작업이 있습니다.`,
          text: "다시 시작 하시겠습니까?",
          icon: "warning",
          showCancelButton: true,
          confirmButtonColor: "#3085d6",
          cancelButtonColor: "#d33",
          confirmButtonText: "확인",
          cancelButtonText: "취소"
        }).then(async result => {
          if (result.isConfirmed) {
            await updateProjectType(this.projectInfo.id, ProjectType.AUTO);
            await this.containerStartRequest("bms");
            this.$emit("restart", "bms");
          }
        });
      } else {
        await updateProjectType(this.projectInfo.id, ProjectType.AUTO);
        await this.containerStartRequest("bms");
        this.$emit("restart", "bms");
      }

      this.SET_PROJECT({
        project_type: ProjectType.AUTO
      });
    },

    async menualCreate() {
      // 실행중인 컨테이너가 있다면 종료
      if (this.projectInfo?.container && this.projectInfo?.container !== "" && this.projectInfo?.container !== "init") {
        Swal.fire({
          title: `실행 중인 작업이 있습니다.`,
          text: "다시 시작 하시겠습니까?",
          icon: "warning",
          showCancelButton: true,
          confirmButtonColor: "#3085d6",
          cancelButtonColor: "#d33",
          confirmButtonText: "확인",
          cancelButtonText: "취소"
        }).then(async result => {
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

    // showContainerName(container) {
    //   if (container) {
    //     if (container.toLowerCase() === "bms") {
    //       return "BMS";
    //     } else if (container.toLowerCase() === "yoloe") {
    //       return "Auto NN";
    //     } else if (container.toLowerCase() === "codegen") {
    //       return "Code Gen";
    //     } else if (container.toLowerCase() === "imagedeploy") {
    //       return "Image Deploy";
    //     } else {
    //       return "";
    //     }
    //   } else {
    //     return "";
    //   }
    // },

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
      this.updateLog({ message: res.message });
      this.updateLog({ message: res.response });
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

<style>
.log-area textarea {
  min-height: 400px;
}
</style>

<style>
.custom .v-banner__wrapper {
  padding: 4px !important;
}
</style>
