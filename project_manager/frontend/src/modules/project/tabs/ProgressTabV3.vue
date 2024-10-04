<template>
  <div class="d-flex flex-column" style="gap: 15px; margin-left: -16px">
    <div>
      <div style="width: 100%" class="d-flex justify-center">
        <div style="gap: 15px" class="d-flex align-center">
          <v-btn
            class="mb-3"
            :color="projectInfo?.project_type === ProjectType.MANUAL ? '#4a80ff' : '#ddd'"
            width="380"
            dark
            @click="manualCreate"
            :disabled="projectInfo?.container_status === 'started' || projectInfo?.container_status === 'running'"
          >
            Manually Generation of Neural Networks
          </v-btn>

          <v-btn
            class="mb-3"
            :color="projectInfo?.project_type === ProjectType.AUTO ? '#4a80ff' : '#ddd'"
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
      <v-card color="#DFDFDF" class="ma-1" style="border-radius: 4px" height="230">
        <div v-if="isNextVersion" style="width: 100%" class="d-flex justify-center">
          <div :style="{ width: isEditing ? '60%' : '53%' }">
            <div style="width: 100%; height: 60px" class="d-flex justify-center align-center">
              <v-btn color="tango" dark :disabled="!isStartNextPipeline" @click="onNextPipeline">
                next version dataset iteration
              </v-btn>
            </div>
            <div style="border: 3px dashed black; height: 30px; border-bottom: none" class="d-flex align-end">
              <div style="width: 3px; height: 3px" class="arrow-head"></div>
            </div>
          </div>
        </div>
        <div :style="{ height: isNextVersion ? 'calc(100% - 90px)' : '100%' }">
          <ProgressCanvas
            :running="projectInfo?.container"
            :status="projectInfo?.container_status"
            :userEdit="project.deploy_user_edit === 'yes'"
            :workflow="project?.workflow"
            @start="start"
            @immediateLaunch="immediateLaunch"
          />
          <!-- @showVis2code="showVis2code" -->
        </div>
      </v-card>
    </div>
    <div>
      <LogViewer />
    </div>
  </div>
</template>
<script>
import Swal from "sweetalert2";
import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProgressCanvas from "@/modules/project/components/ProgressCanvas.vue";

import LogViewer from "@/modules/project/log-viewer/LogViewer.vue";

import { ProjectType } from "@/shared/consts";
import { DisplayName, TaskType, ContainerName, LearningType, ProjectStatus } from "@/shared/enums";

import { containerStart, updateProjectType } from "@/api";
export default {
  components: { ProgressCanvas, LogViewer },
  props: {
    projectInfo: {
      default: () => ({})
    }
  },
  data() {
    return {
      running: "",
      isVis2Code: false,
      open: false,
      DisplayName,
      TaskType,
      ContainerName,
      ProjectType
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project", "autonn_status"]),

    isNextVersion() {
      return this.isIncremental && this.project.version < 5;
    },

    isIncremental() {
      return this.project?.learning_type === LearningType.INCREMENTAL;
    },

    isEditing() {
      return this.project.workflow?.length >= 4;
    },

    isStartNextPipeline() {
      const workflowLength = this.project.workflow?.length;
      if (!workflowLength) return false;

      const currentWorkflowName = this.project.workflow?.[workflowLength - 1]?.workflow_name;
      if (!currentWorkflowName) return false;

      const isCompleted = this.project?.container_status === ProjectStatus.COMPLETED;

      return this.isIncremental && currentWorkflowName === this.project.container && isCompleted;
    }
  },

  mounted() {},

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT
    }),

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

          this.$emit("start");
          // await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
          // this.SET_PROJECT({ project_type: ProjectType.MANUAL });
          await this.containerStartRequest(container);
        }
      });
    },

    async immediateLaunch(container) {
      this.$emit("start");
      await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
      await this.containerStartRequest(container);
      this.SET_PROJECT({ project_type: ProjectType.MANUAL, container: container });
    },

    async autoCreate() {
      await updateProjectType(this.projectInfo.id, ProjectType.AUTO);
      this.SET_PROJECT({ project_type: ProjectType.AUTO });

      if (this.projectInfo?.container_status === ProjectStatus.COMPLETED) {
        const curIndex = this.projectInfo?.workflow?.findIndex(q => q.workflow_name === this.projectInfo.container);
        if (curIndex < 0) return;

        const containerName = this.projectInfo?.workflow?.[curIndex + 1]?.workflow_name;
        if (!containerName) return;

        this.SET_PROJECT({ container: containerName, container_status: ProjectStatus.STARTED });
        this.$emit("start");
        await this.containerStartRequest(containerName);
      } else if (
        this.projectInfo?.container_status !== ProjectStatus.RUNNING &&
        this.projectInfo?.container_status !== ProjectStatus.STARTED
      ) {
        if (!this.projectInfo?.container || this.projectInfo?.container === "init") {
          const workflow = this.projectInfo?.workflow?.[0]?.workflow_name;
          const containerName = workflow || ContainerName.AUTO_NN;
          this.SET_PROJECT({ container: containerName, container_status: ProjectStatus.STARTED });
          this.$emit("start");
          await this.containerStartRequest(containerName);
        } else {
          this.SET_PROJECT({ container: this.projectInfo?.container, container_status: ProjectStatus.STARTED });
          this.$emit("start");
          await this.containerStartRequest(this.projectInfo?.container);
        }
      }
    },

    async manualCreate() {
      await updateProjectType(this.projectInfo.id, ProjectType.MANUAL);
      this.SET_PROJECT({ project_type: ProjectType.MANUAL });
    },

    async containerStartRequest(container) {
      const res = await containerStart(container, this.projectInfo.create_user, this.projectInfo.id);
      this.$EventBus.$emit("logUpdate", { message: res.message });
      this.$EventBus.$emit("logUpdate", { message: res.response });
    },

    onNextPipeline() {
      this.$emit("nextPipeline");
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

<style>
.arrow-head::after {
  height: 16px;
  width: 3px;
  background-color: black;
  transform-origin: top center;
  transform: translateY(-14px) translateX(-3px) rotate(-135deg);

  content: " ";
  display: block;
}
.arrow-head::before {
  height: 16px;
  width: 3px;
  background-color: black;
  transform-origin: top center;
  transform: translateY(2px) translateX(-3px) rotate(135deg);

  content: " ";
  display: block;
}
</style>
