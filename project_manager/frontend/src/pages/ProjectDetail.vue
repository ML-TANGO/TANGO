<template lang="">
  <div>
    <v-card style="height: calc(100vh - 60px); overflow-y: overlay; width: 100%" class="px-4">
      <TabBase :count="-1" title="Configuration">
        <template #action>
          <ProjectCreateDialog :step="step" @stepChange="onStepChange">
            <template v-slot:btn>
              <v-btn icon class="ml-3 mr-5" @click="onEdit"><v-icon>mdi-pencil-outline</v-icon></v-btn>
            </template>
          </ProjectCreateDialog>
        </template>
        <template #content> <ConfigurationTab :projectInfo="projectInfo" /> </template>
      </TabBase>
      <div>
        <TabBase :count="-1" title="Progress">
          <template #content> <ProgressTab :projectInfo="projectInfo" @restart="restart" @start="start" /> </template>
        </TabBase>
      </div>
    </v-card>
    <KaggleUserInfoDialog ref="KaggleUserInfoDialogref" @restart="isAlreadyDataset"></KaggleUserInfoDialog>
    <v-dialog v-model="isOpenDownloadingDialog" width="450" persistent>
      <v-card style="text-align: center">
        <div style="padding: 30px">
          <v-progress-circular indeterminate color="primary" size="85" width="8"></v-progress-circular>
        </div>
        <v-card-text style="font-size: 16px">
          "{{ projectInfo?.dataset || "" }}" Dataset을 다운로드 중입니다.
        </v-card-text>
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
import ProgressTab from "@/modules/project/tabs/ProgressTabV2.vue";
import TabBase from "@/modules/project/TabBase.vue";
import KaggleUserInfoDialog from "@/modules/common/dialog/KaggleUserInfoDialog.vue";

import { /*TaskType,*/ ContainerName, CommonDatasetName } from "@/shared/enums";

import {
  getProjectInfo,
  getTargetInfo,
  getDatasetListTango,
  stopContainer,
  postStatusRequest,
  setWorkflow,
  checkExistKaggleInfo,
  imagenetDatasetDownload,
  cocoDatasetDownload,
  vocDatasetDownload,
  kaggleDatasetDownload
} from "@/api";

import Cookies from "universal-cookie";
export default {
  components: { ConfigurationTab, ProgressTab, TabBase, ProjectCreateDialog, KaggleUserInfoDialog },
  data() {
    return {
      step: 1,
      projectInfo: null,
      interval: null,
      isOpenDownloadingDialog: false,
      datasetDownloadingInterval: null
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"])
  },

  async mounted() {
    const cookie_info = new Cookies();
    const user_info = cookie_info.get("userinfo");
    try {
      this.projectInfo = await getProjectInfo(this.$route.params.id);
      this.SET_PROJECT(this.projectInfo);
      if (this.projectInfo.create_user !== user_info) {
        Swal.fire("잘못된접근입니다.");
        this.$router.push("/");
      }
      let status = true;
      const info = this.projectInfo;

      if (!info?.dataset || info?.dataset === "") status = false;
      else if (!info?.target_id || !info?.target_info) status = false;
      else if (!info?.target_id || info?.target_id === "") status = false;
      else if (!info?.task_type || info?.task_type === "") status = false;
      else if (!info?.nas_type || info?.nas_type === "") status = false;
      else if (!info?.deploy_weight_level || info?.deploy_weight_level === "") status = false;
      else if (!info?.deploy_precision_level || info?.deploy_precision_level === "") status = false;
      else if (!info?.deploy_user_edit || info?.deploy_user_edit === "") status = false;
      else if (!info?.deploy_output_method || info?.deploy_output_method === "") status = false;
      if (status === false) {
        Swal.fire("project를 완성해 주세요.");
        this.$router.push("/");
        return;
      }

      if (info.container_status === "running" || info.container_status === "started") {
        this.startInterval();
        if (info.container === ContainerName.VISUALIZATION) {
          this.$EventBus.$emit("control_Vis2Code", true);
        }
      }

      if (!info?.workflow || info?.workflow.length <= 0) {
        // 20240610........ 이전 버전...................
        // const workflow =
        // info.task_type === TaskType.DETECTION
        //   ? [ContainerName.BMS, ContainerName.AUTO_NN, ContainerName.CODE_GEN, ContainerName.IMAGE_DEPLOY]
        //   : [
        //       ContainerName.BMS,
        //       ContainerName.VISUALIZATION,
        //       ContainerName.AUTO_NN_RESNET,
        //       ContainerName.CODE_GEN,
        //       ContainerName.IMAGE_DEPLOY
        //     ];

        // info.task_type === TaskType.DETECTION
        //   ? [ContainerName.AUTO_NN, ContainerName.CODE_GEN, ContainerName.IMAGE_DEPLOY]
        //   : [
        //       // ContainerName.VISUALIZATION,
        //       ContainerName.AUTO_NN_RESNET,
        //       ContainerName.CODE_GEN,
        //       ContainerName.IMAGE_DEPLOY
        //     ];

        const workflow = [ContainerName.AUTO_NN, ContainerName.CODE_GEN, ContainerName.IMAGE_DEPLOY];

        if (info.deploy_user_edit === "yes") {
          workflow.splice(workflow.length - 1, 0, ContainerName.USER_EDITING);
        }

        const res = await setWorkflow(info.id, workflow);

        this.projectInfo = {
          ...this.projectInfo,
          workflow: res.workflow
        };
        this.SET_PROJECT(this.projectInfo);
      }

      this.isAlreadyDatasetHandler();
    } catch {
      Swal.fire("잘못된접근입니다.");
      this.$router.push("/");
    }
  },

  destroyed() {
    this.stopInterval();
    if (this.datasetDownloadingInterval) {
      clearInterval(this.datasetDownloadingInterval);
    }
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT,
      SET_SELECTED_TARGET: ProjectMutations.SET_SELECTED_TARGET,
      SET_SELECTED_IMAGE: ProjectMutations.SET_SELECTED_IMAGE
    }),

    startInterval() {
      if (this.interval === null) {
        this.interval = setInterval(async () => {
          if (this.projectInfo.id) {
            await postStatusRequest({ user_id: this.projectInfo.create_user, project_id: this.projectInfo.id }).then(
              res => {
                if (res === null) return;
                if (typeof res !== "string") {
                  this.SET_PROJECT({ container: res.container, container_status: res.container_status });
                  this.projectInfo = this.project;
                  this.$EventBus.$emit("logUpdate", res);

                  if (res.container_status.toLowerCase() === "failed") {
                    this.stopInterval();

                    return;
                  }

                  if (res.container === ContainerName.VISUALIZATION) {
                    if (res.container_status !== "running" && res.container_status !== "started") {
                      // VISUALIZATION 완료 경우
                      this.$EventBus.$emit("control_Vis2Code", false);
                    }
                  }

                  if (this.projectInfo.project_type !== "auto") {
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
                }
              }
            );
          }
        }, 5000); // 5s
      }
    },

    stopInterval() {
      if (this.interval) {
        clearInterval(this.interval);
        this.interval = null;
      }
    },

    start() {
      this.SET_PROJECT({ container: "", container_status: "" });
      this.projectInfo = this.project;

      this.startInterval();
    },

    restart(container) {
      this.SET_PROJECT({ container: container, container_status: "started" });
      this.projectInfo = this.project;

      this.startInterval();
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
          if (this.projectInfo?.target_id && this.projectInfo.target_id !== "") {
            const targetInfo = await getTargetInfo(this.projectInfo.target_id);
            this.SET_SELECTED_TARGET(targetInfo);
          }

          //선택한 dataset 정보
          if (this.projectInfo?.dataset && this.projectInfo.dataset !== "") {
            const datasetList = await getDatasetListTango();
            const datasetInfo = datasetList.find(q => q.name === this.projectInfo.dataset);
            if (datasetInfo) this.SET_SELECTED_IMAGE(datasetInfo);
          }

          // 실행중인 컨테이너가 있다면 종료
          if (
            this.projectInfo?.container &&
            this.projectInfo?.container !== "" &&
            this.projectInfo?.container !== "init"
          ) {
            //Todo container 상태 추가
            await stopContainer(this.projectInfo.container, this.projectInfo.create_user, this.projectInfo.id);
          }
        } else {
          this.$EventBus.$emit("forcedTermination");
        }
      });
    },

    isAlreadyDatasetHandler() {
      this.isAlreadyDataset();

      const time = 60 * 30 * 1000;
      this.datasetDownloadingInterval = setInterval(() => {
        this.isAlreadyDataset();
      }, time);
    },

    async isAlreadyDataset() {
      const datasetDownloadType = Object.freeze({
        COMPLETE: 0x01,
        DOWNLOADING: 0x10,
        BEFORE_DOWNLOADING: 0x20
      });

      if (!Object.values(CommonDatasetName).includes(this.projectInfo.dataset)) {
        return true;
      }

      const checkIsAlready = value => {
        if (value["isAlready"] === true) {
          return datasetDownloadType.DOWNLOADING;
        } else if (value["complete"] === true) {
          return datasetDownloadType.COMPLETE;
        }
        return datasetDownloadType.BEFORE_DOWNLOADING;
      };

      let result = datasetDownloadType.BEFORE_DOWNLOADING;

      if (this.projectInfo.dataset === CommonDatasetName.IMAGE_NET) {
        const res = await imagenetDatasetDownload();
        result = checkIsAlready(res);
      } else if (this.projectInfo.dataset === CommonDatasetName.CHESTXRAY) {
        const checkKaggleInfo = await checkExistKaggleInfo();
        if (!checkKaggleInfo.isExist) {
          this.$refs.KaggleUserInfoDialogref.isOpen = true;
          return;
        }
        const res = await kaggleDatasetDownload();
        result = checkIsAlready(res);
      } else if (this.projectInfo.dataset === CommonDatasetName.VOC) {
        const res = await vocDatasetDownload();
        result = checkIsAlready(res);
      } else if (this.projectInfo.dataset === CommonDatasetName.COCO) {
        const res = await cocoDatasetDownload();
        result = checkIsAlready(res);
      }

      if (result !== datasetDownloadType.COMPLETE) {
        this.isOpenDownloadingDialog = true;
      } else {
        this.isOpenDownloadingDialog = false;
        if (this.datasetDownloadingInterval) {
          clearInterval(this.datasetDownloadingInterval);
        }
      }
    },

    onBack() {
      this.$router.push("/");
    }
  }
};
</script>
<style lang="scss"></style>
