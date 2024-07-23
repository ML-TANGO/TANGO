<template lang="">
  <div>
    <v-card style="height: calc(100vh - 60px); overflow-y: overlay; width: 100%" class="px-4">
      <TabBase :count="-1" title="Configuration" ref="ConfigurationRef">
        <template #action>
          <ProjectCreateDialog :step="step" @stepChange="onStepChange">
            <template v-slot:btn>
              <v-btn icon class="ml-3 mr-5" @click="onEdit"><v-icon>mdi-pencil-outline</v-icon></v-btn>
            </template>
          </ProjectCreateDialog>
        </template>
        <template #content> <ConfigurationTab :projectInfo="project" /> </template>
      </TabBase>
      <div>
        <TabBase :count="-1" title="Progress" :defaultBanner="true" ref="ProgressRef">
          <template #content> <ProgressTab :projectInfo="project" @restart="restart" @start="start" /> </template>
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

import ProjectCreateDialog from "@/modules/project/ProjectCreateDialog.vue";
import ConfigurationTab from "@/modules/project/tabs/ConfigurationTab.vue";
import ProgressTab from "@/modules/project/tabs/ProgressTabV3.vue";
import TabBase from "@/modules/project/TabBase.vue";
// import KaggleUserInfoDialog from "@/modules/common/dialog/KaggleUserInfoDialog.vue";

import { Project } from "@/shared/models";
import { DatasetStatus, ContainerName } from "@/shared/enums";

import { getProjectInfo, get_autonn_status, postStatusRequest } from "@/api";

export default {
  components: { ConfigurationTab, ProgressTab, TabBase, ProjectCreateDialog /*, KaggleUserInfoDialog */ },
  data() {
    return {
      step: 1,
      projectInfo: null,
      isOpenDownloadingDialog: false,
      datasetDownloadingInterval: null,

      projectStatusInterval: null
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"])
  },

  async mounted() {
    try {
      await this.getProjectInfo();
      get_autonn_status(this.project.id).then(res => {
        this.SET_AUTO_NN_STATUS(res.autonn);
      });

      if (this.project.container_status === "running" || this.project.container_status === "started") {
        this.startProjectStatusInterval();
      }
    } catch (err) {
      this.denyAccess();
    }
  },

  destroyed() {
    if (this.datasetDownloadingInterval) clearInterval(this.datasetDownloadingInterval);
    if (this.projectStatusInterval) clearInterval(this.projectStatusInterval);
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT,
      SET_SELECTED_TARGET: ProjectMutations.SET_SELECTED_TARGET,
      SET_SELECTED_IMAGE: ProjectMutations.SET_SELECTED_IMAGE,
      SET_AUTO_NN_STATUS: ProjectMutations.SET_AUTO_NN_STATUS
    }),

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
      console.log("response", response);
      const projectInfo = new Project(response);

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

    async onEdit() {},

    restart() {
      this.startProjectStatusInterval();
    },

    start() {},

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
      if (this.projectStatusInterval) return;

      // this.getCurrentProjectInfo();
      this.projectStatusInterval = setInterval(() => {
        this.getCurrentProjectInfo();
      }, 5000); //5s
    },

    async getCurrentProjectInfo() {
      get_autonn_status(this.project.id).then(res => {
        this.SET_AUTO_NN_STATUS(res.autonn);
        // if (res.autonn?.project) {
        //   this.SET_PROJECT(res.autonn?.project);
        // }
      });

      if (this.project) {
        await postStatusRequest({ user_id: this.project.create_user, project_id: this.project.id }).then(res => {
          if (res === null) return;
          if (typeof res === "string") return;

          this.SET_PROJECT({ container: res.container, container_status: res.container_status });
          this.$EventBus.$emit("logUpdate", res);
          if (res.container_status.toLowerCase() === "failed") {
            this.stopInterval();
            return;
          }

          if (this.project.project_type !== "auto") {
            if (res.container_status !== "running" && res.container_status !== "started") {
              this.stopInterval();
              if (res.container === ContainerName.IMAGE_DEPLOY) {
                this.$EventBus.$emit("nnModelDownload");
              }
              return;
            }
          } else {
            // todo auto일경우 구현
            if (res.container === ContainerName.IMAGE_DEPLOY) {
              if (res.container_status !== "running" && res.container_status !== "started") {
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
    }
  }
};
</script>
<style lang="scss"></style>
