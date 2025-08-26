<template lang="">
  <div>
    <v-card style="height: calc(100vh - 60px); overflow-y: overlay; width: 100%" class="px-4">
      <TabBase :count="-1" title="Configuration" ref="ConfigurationRef">
        <template #action>
          <ProjectCreateDialog :step="step" @stepChange="onStepChange" @close="onClose">
            <template v-slot:btn>
              <v-btn icon class="ml-3 mr-5" @click="onEdit"><v-icon>mdi-pencil-outline</v-icon></v-btn>
            </template>
          </ProjectCreateDialog>
        </template>
        <template #content="{ isOpen }"> <ConfigurationTab :projectInfo="project" :isOpen="isOpen" /> </template>
      </TabBase>
      <div>
        <TabBase :count="-1" title="Progress" :defaultBanner="true" ref="ProgressRef">
          <template #content>
            <ProgressTab :projectInfo="project" @start="start" @nextPipeline="nextPipeline" @stop="stop" />
          </template>
        </TabBase>
      </div>
    </v-card>

    <v-dialog v-model="isOpenDownloadingDialog" width="450" persistent>
      <v-card style="text-align: center">
        <div style="padding: 30px">
          <v-progress-circular indeterminate color="primary" size="85" width="8"></v-progress-circular>
        </div>
        <v-card-text style="font-size: 16px"> "{{ project?.dataset || "" }}" Dataset을 다운로드 중입니다. </v-card-text>
        <v-card-text style="font-size: 16px"> 잠시만 기다려 주세요. </v-card-text>
        <v-card-actions class="justify-center">
          <v-btn dark color="#4a80ff" @click="onBack">다른 프로젝트 보기</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
<script>
import Swal from "sweetalert2";
import { mapState, mapMutations } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProjectCreateDialog from "@/modules/project/ProjectCreateDialogV2.vue";
import ConfigurationTab from "@/modules/project/tabs/ConfigurationTab.vue";
import ProgressTab from "@/modules/project/tabs/ProgressTabV3.vue";
import TabBase from "@/modules/project/TabBase.vue";
// import KaggleUserInfoDialog from "@/modules/common/dialog/KaggleUserInfoDialog.vue";

import { Project } from "@/shared/models";
import { DatasetStatus, ContainerName, ProjectStatus, TaskType } from "@/shared/enums";

import {
  getProjectInfo,
  get_autonn_status,
  postStatusRequest,
  stopContainer,
  getTargetInfo,
  getDatasetListTango,
  getUserIntervalTime,
  next_pipeline_start
} from "@/api";

export default {
  components: { ConfigurationTab, ProgressTab, TabBase, ProjectCreateDialog /*, KaggleUserInfoDialog */ },
  data() {
    return {
      step: 1,
      projectInfo: null,
      isOpenDownloadingDialog: false,
      datasetDownloadingInterval: null,

      projectStatusInterval: null,
      projectStatusIntervalTime: 20,
      autonnStatusInterval: null,
      autonnStatusIntervalTime: 20
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"])
  },

  async mounted() {
    try {
      await this.getProjectInfo();
      get_autonn_status(this.project.id).then(res => {
        if (res) {
          let response = null;

          if (typeof res.replace === typeof "") response = JSON.parse(res.replace(/Infinity/g, '"Infinity"'));
          else response = res;
          if (response) this.SET_AUTO_NN_STATUS(response.autonn);
        }
      });

      const intervalTimes = await getUserIntervalTime();

      this.projectStatusIntervalTime = intervalTimes["project_status"];
      this.autonnStatusIntervalTime = intervalTimes["autonn_status"];

      if (
        this.project.container_status === ProjectStatus.RUNNING ||
        this.project.container_status === ProjectStatus.STARTED
      ) {
        this.startProjectStatusInterval();
        setTimeout(() => {
          this.getCurrentProjectInfo();
        }, 2000);
      }

      // if (
      //   this.project?.container === ContainerName.IMAGE_DEPLOY &&
      //   this.project?.container_status === ProjectStatus.COMPLETED
      // ) {
      //   await this.nextPipeline();
      // }
    } catch (err) {
      this.denyAccess();

      return;
    }

    this.$EventBus.$on("updateProgressTime", time => {
      this.projectStatusIntervalTime = time;
      if (this.projectStatusInterval) {
        this.stopInterval();
        this.startProjectStatusInterval();
      }
    });

    this.$EventBus.$on("updateAutonnTime", time => {
      this.autonnStatusIntervalTime = time;

      if (this.autonnStatusInterval) {
        this.stopInterval();
        this.startProjectStatusInterval();
      }
    });
  },

  destroyed() {
    if (this.datasetDownloadingInterval) clearInterval(this.datasetDownloadingInterval);
    if (this.projectStatusInterval) clearInterval(this.projectStatusInterval);
    if (this.autonnStatusInterval) clearInterval(this.autonnStatusInterval);
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT,
      SET_SELECTED_TARGET: ProjectMutations.SET_SELECTED_TARGET,
      SET_SELECTED_IMAGE: ProjectMutations.SET_SELECTED_IMAGE,
      SET_AUTO_NN_STATUS: ProjectMutations.SET_AUTO_NN_STATUS
    }),

    onClose() {
      window.location.reload();
    },

    denyAccess(title = "잘못된접근입니다.") {
      Swal.fire(title);
      this.$router.push("/");
    },

    async getProjectInfo() {
      const projectId = this.$route.params.id;
      const response = await getProjectInfo(projectId);

      if (!response) {
        this.denyAccess();
        return;
      }

      await this.setProjectInfo(response);
    },

    async setProjectInfo(data) {
      const projectInfo = new Project(data);

      if (!projectInfo.validation()) {
        this.denyAccess();
        return;
      }

      await projectInfo.load();

      this.checkDataset(projectInfo);

      this.SET_PROJECT(projectInfo);
    },

    onStepChange(step) {
      this.step = step;
    },

    async onEdit() {
      Swal.fire({
        title: `프로젝트를 수정 하시겠습니까?`,
        text: "지금까지 진행된 내용이 사라집니다.",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "확인",
        cancelButtonText: "취소"
      }).then(async result => {
        if (result.isConfirmed) {
          //선택한 target정보
          if (this.project?.target_id && this.project.target_id !== "") {
            const targetInfo = await getTargetInfo(this.project.target_id);
            this.SET_SELECTED_TARGET(targetInfo);
          }

          //선택한 dataset 정보
          if (this.project?.dataset && this.project.dataset !== "") {
            const datasetList = await getDatasetListTango();
            const datasetInfo = datasetList.find(q => q.name === this.project.dataset);
            if (datasetInfo) this.SET_SELECTED_IMAGE(datasetInfo);
          }

          // 실행중인 컨테이너가 있다면 종료
          if (this.project?.container && this.project?.container !== "" && this.project?.container !== "init") {
            //Todo container 상태 추가
            await stopContainer(this.project.container, this.project.create_user, this.project.id);
          }
        } else {
          this.$EventBus.$emit("forcedTermination");
        }
      });
    },

    stop() {
      this.stopInterval();
    },

    start() {
      this.startProjectStatusInterval();

      setTimeout(() => {
        this.getCurrentProjectInfo();
      }, 2000);
    },

    onBack() {
      this.$router.push("/");
    },

    checkDataset(projectInfo) {
      if (projectInfo.datasetObject?.status === DatasetStatus.DOWNLOADING) {
        this.isOpenDownloadingDialog = true;

        const time = 30 * 1000; // 30s
        this.datasetDownloadingInterval = setInterval(async () => {
          await projectInfo.load();
          if (projectInfo.datasetObject?.status === DatasetStatus.COMPLETE) {
            clearInterval(this.datasetDownloadingInterval);
            this.datasetDownloadingInterval = null;
            this.isOpenDownloadingDialog = false;
          }
        }, time);
      }
    },

    // =============================================================================
    startProjectStatusInterval() {
      if (this.project.container === ContainerName.AUTO_NN && this.project.task_type !== TaskType.CHAT) {
        if (!this.autonnStatusInterval) {
          this.autonnStatusInterval = setInterval(() => {
            this.getCurrentAutonnStatus();
          }, this.secondToMillisecond(this.autonnStatusIntervalTime)); //1s
        }
      }

      if (this.projectStatusInterval) return;
      this.projectStatusInterval = setInterval(() => {
        this.getCurrentProjectInfo();
      }, this.secondToMillisecond(this.projectStatusIntervalTime)); //10s
    },

    async getCurrentAutonnStatus() {
      get_autonn_status(this.project.id).then(res => {
        this.SET_AUTO_NN_STATUS(res.autonn);
      });
    },

    async getCurrentProjectInfo() {
      if (this.project) {
        await postStatusRequest({ user_id: this.project.create_user, project_id: this.project.id }).then(async res => {
          if (res === null) return;
          // 오류가 발생하여 massage로 반환된 경우
          if (typeof res === "string") return;

          this.SET_PROJECT({ container: res.container, container_status: res.container_status });
          this.$EventBus.$emit("logUpdate", res);
          if (res.container_status.toLowerCase() === ProjectStatus.FAILED) {
            this.stopInterval();
            return;
          }

          console.log("res.container_status.toLowerCase()", res.container_status.toLowerCase());
          if (res.container_status.toLowerCase() === ProjectStatus.STOPPED) {
            this.stopInterval();
            return;
          }

          if (this.project.project_type !== "auto") {
            if (res.container_status !== ProjectStatus.RUNNING && res.container_status !== ProjectStatus.STARTED) {
              this.stopInterval();
              if (res.container === ContainerName.IMAGE_DEPLOY) {
                this.$EventBus.$emit("nnModelDownload");
              }
              return;
            }
          } else {
            // todo auto일경우 구현
            if (res.container === ContainerName.IMAGE_DEPLOY) {
              if (res.container_status !== ProjectStatus.RUNNING && res.container_status !== ProjectStatus.STARTED) {
                this.stopInterval();
                this.$EventBus.$emit("nnModelDownload");
                return;
              }
            }
          }
        });
      }
    },

    stopInterval() {
      clearInterval(this.projectStatusInterval);
      this.projectStatusInterval = null;

      clearInterval(this.autonnStatusInterval);
      this.autonnStatusInterval = null;
    },

    secondToMillisecond(second) {
      return second * 1000;
    },

    async nextPipeline() {
      const result = await Swal.fire({
        title: `진행`,
        text: "다음 파이프라인을 실행 하시겠습니까?",
        icon: "info",
        showCancelButton: true,
        allowOutsideClick: false,
        allowEscapeKey: false,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "확인",
        cancelButtonText: "취소"
      });

      if (!result.isConfirmed) return;

      const response = await next_pipeline_start(this.project.create_user, this.project.id);
      await this.setProjectInfo(response.project);

      this.startProjectStatusInterval();
      setTimeout(() => {
        this.getCurrentProjectInfo();
      }, 2000);
    }
  }
};
</script>
<style lang="scss"></style>
