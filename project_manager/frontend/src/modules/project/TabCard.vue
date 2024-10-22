<template>
  <v-hover v-slot="{ hover }">
    <div style="width: 360px; height: 230px; perspective: 1200px; position: relative">
      <v-card style="border: 1px solid #4a80ff" class="box-shadow flip" @click="navigation">
        <!--  :style="{ transform: hover ? 'rotateY(-180deg)' : 'rotateY(0)' }"    -->
        <div class="front">
          <v-list-item class="my-3" style="padding: 0px 0px 0px 16px">
            <v-list-item-content style="width: 300px">
              <v-list-item-title class="text-subtitle-1 font-weight-bold">
                {{ projectInfo.project_name }}
              </v-list-item-title>
              <v-list-item-subtitle style="font-size: 12px">
                {{ projectInfo.project_description }}
              </v-list-item-subtitle>
            </v-list-item-content>
            <v-list-item-action>
              <div
                :style="{ border: `1px solid ${status.color}`, color: status.color, borderRight: 'none' }"
                style="font-size: 12px; width: 100%; text-align: center; border-radius: 15px 0 0 15px"
                class="px-1 d-flex"
              >
                <div style="padding: 2px; min-width: 85px">
                  <v-icon v-if="status.color === '#FF3D54'" color="#FF3D54" size="15" class="mr-1">
                    mdi-alert-circle-outline
                  </v-icon>
                  {{ status.title }}
                </div>
              </div>
            </v-list-item-action>
          </v-list-item>

          <v-divider />

          <div class="d-flex flex-column pt-2" style="width: 100%; gap: 5px; height: calc(100% - 90px)">
            <!-- information -->
            <div class="d-flex flex-column px-4" style="width: 100%; height: calc(100% - 40px); gap: 3px">
              <div class="d-flex" v-for="(item, index) in itemBasicinfos" :key="index">
                <div style="width: 40%; text-align: start; font-size: 12px; color: black">{{ item.title }}</div>
                <div
                  v-if="item.title !== 'Target'"
                  style="width: 60%; text-align: end; font-size: 12px; color: #00000099"
                >
                  {{ item.content }}
                </div>
                <div v-else style="width: 60%; text-align: end; font-size: 12px; color: #00000099">
                  {{ item.content?.target_name }}
                </div>
              </div>
            </div>
            <!-- footer action -->
            <div style="height: 40px; gap: 8px" class="d-flex justify-center align-center">
              <slot name="action">
                <ProjectCreateDialog :step="step" @stepChange="onStepChange" @close="close">
                  <template v-slot:btn>
                    <v-btn height="30" style="width: 330px" outlined color="#FF3D54" @click="setupBtn">
                      To Set Up →
                    </v-btn>
                  </template>
                </ProjectCreateDialog>
              </slot>
            </div>
          </div>
        </div>

        <div class="back d-flex align-center justify-space-around" style="width: 360px"></div>
      </v-card>
      <v-fab-transition leave-absolute>
        <div
          v-if="hover"
          class="px-2"
          style="
            position: absolute;
            top: -15px;
            right: 10px;
            background-color: white;
            border: 1px solid #4a80ff;
            border-radius: 8px;
            transition: 3s;
          "
        >
          <v-btn icon height="30" @click="onDelete">
            <v-icon size="20" color="#ff3d54">mdi-delete</v-icon>
          </v-btn>
        </div>
      </v-fab-transition>
    </div>
  </v-hover>
</template>
<script>
import Swal from "sweetalert2";
import { mapMutations } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProjectCreateDialog from "@/modules/project/ProjectCreateDialogV2.vue";

import { deleteProject, getTargetInfo, getDatasetListTango } from "@/api";
import { TaskType } from "@/shared/enums";
export default {
  components: { ProjectCreateDialog },

  props: {
    status: {
      default: () => ({
        title: "REDAY",
        color: "#4a80ff"
      })
    },

    projectInfo: {
      default: () => ({
        project_name: "",
        project_description: "",
        dataset: "",
        target: "",
        task_type: "",
        nas_type: ""
      })
    }
  },

  data() {
    return {
      step: 1,
      itemBasicinfosColumn: [
        { title: "Dataset", value: "dataset" },
        { title: "Target", value: "target_info" },
        { title: "Task Type", value: "task_type" }
      ],
      itemDetailInfosColumn: ["autonn_dataset_file", "autonn_basemodel"] // 등등 추가 예정
    };
  },

  computed: {
    itemBasicinfos() {
      return this.itemBasicinfosColumn.map(q => ({ title: q.title, content: this.projectInfo[q.value] }));
    },

    itemDetailInfos() {
      return this.itemBasicinfosColumn.map(q => ({ title: q, content: this.projectInfo[q] }));
    }
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT,
      SET_SELECTED_TARGET: ProjectMutations.SET_SELECTED_TARGET,
      SET_SELECTED_IMAGE: ProjectMutations.SET_SELECTED_IMAGE,
      INIT_PROJECT: ProjectMutations.INIT_PROJECT
    }),

    async setupBtn() {
      this.SET_PROJECT(this.projectInfo);
      if (this.projectInfo.container === "" || this.projectInfo.container === "init") {
        await this.setDialog();
      } else {
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
            await this.setDialog();
          } else {
            this.$EventBus.$emit("forcedTermination");
          }
        });
      }
    },

    navigation() {
      let status = true;
      const info = this.projectInfo;
      if (info.task_type !== TaskType.CHAT && (!info?.dataset || info?.dataset === "")) status = false;
      else if (!info?.target_id || !info?.target_info) status = false;
      else if (!info?.target_id || info?.target_id === "") status = false;
      else if (!info?.task_type || info?.task_type === "") status = false;
      else if (!info?.nas_type || info?.nas_type === "") status = false;
      else if (!info?.deploy_weight_level || info?.deploy_weight_level === "") status = false;
      else if (!info?.deploy_precision_level || info?.deploy_precision_level === "") status = false;
      else if (!info?.deploy_user_edit || info?.deploy_user_edit === "") status = false;
      else if (!info?.deploy_output_method || info?.deploy_output_method === "") status = false;

      if (status) {
        this.$router.push(`/project/${this.projectInfo.id}`);
      } else {
        Swal.fire("project를 완성해 주세요.");
      }
    },

    onDelete() {
      Swal.fire({
        title: `${this.projectInfo.project_name} 프로젝트를 \n 삭제하시겠습니까?`,
        text: "삭제한 뒤 복구가 불가능합니다.",
        icon: "warning",
        showCancelButton: true,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "확인",
        cancelButtonText: "취소"
      }).then(async result => {
        if (result.isConfirmed) {
          await deleteProject(this.projectInfo.id);
          this.$EventBus.$emit("deleteProject");
        }
      });
    },

    close() {
      this.$EventBus.$emit("projectDialogclose");
      this.INIT_PROJECT();
    },

    onStepChange(step) {
      this.step = step;
    },

    async setDialog() {
      if (this.projectInfo.project_name === "" || this.projectInfo.project_description === "") {
        this.step = 1;
      } else if (this.status.title === "Dataset") {
        this.step = 2;
      } else if (this.status.title === "Target") {
        this.step = 3;
      } else {
        this.step = 4;
      }

      if (this.projectInfo?.target_id && this.projectInfo.target_id !== "") {
        const targetInfo = await getTargetInfo(this.projectInfo.target_id);
        this.SET_SELECTED_TARGET(targetInfo);
      }

      if (this.projectInfo?.dataset && this.projectInfo.dataset !== "") {
        const datasetList = await getDatasetListTango();
        const datasetInfo = datasetList.find(q => q.name === this.projectInfo.dataset);
        if (datasetInfo) this.SET_SELECTED_IMAGE(datasetInfo);
      }
    }
  }
};
</script>

<style lang="scss" scoped>
.box-shadow {
  -webkit-box-shadow: 6px 5px 12px 1px rgba(0, 0, 0, 0.37) !important;
  box-shadow: 6px 5px 12px 1px rgba(0, 0, 0, 0.37) !important;
}

.flip {
  width: 100%;
  height: 100%;
  position: relative;
  color: white;
  transform-style: preserve-3d;
  transition: 0.5s;
}

.front,
.back {
  width: 100%;
  height: 100%;
  position: absolute;
  backface-visibility: hidden;
}

.back {
  transform: rotateY(180deg);
}
</style>
