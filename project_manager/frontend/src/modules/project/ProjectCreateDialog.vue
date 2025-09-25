<template lang="">
  <v-dialog v-model="dialog" persistent max-width="1100px" scrollable>
    <template v-slot:activator="{ on, attrs }">
      <div v-bind="attrs" v-on="on">
        <slot name="btn"></slot>
      </div>
    </template>
    <v-card>
      <v-card-title>
        <span v-if="project?.id" class="text-h5 font-weight-bold">Edit Project</span>
        <span v-else class="text-h5 font-weight-bold">Create Project</span>
        <v-card-actions>
          <v-btn color="blue darken-1" icon @click="close" absolute right>
            <v-icon color="#ccc" size="34">mdi-close-circle</v-icon>
          </v-btn>
        </v-card-actions>
      </v-card-title>
      <div class="d-flex align-center" style="height: 500px">
        <v-stepper :value="step" vertical class="elevation-0" style="width: 250px; letter-spacing: 1px" non-linear>
          <v-stepper-step :complete="step > 1" step="1">Project Info <small>Enter Project Info</small> </v-stepper-step>
          <v-stepper-content step="1" class="my-3"> </v-stepper-content>

          <v-stepper-step :complete="step > 2" step="2"> Dataset <small>Select Dataset</small> </v-stepper-step>
          <v-stepper-content step="2" class="my-3"></v-stepper-content>

          <v-stepper-step :complete="step > 3" step="3"> Target <small>Select Target</small> </v-stepper-step>
          <v-stepper-content step="3" class="my-3"></v-stepper-content>

          <v-stepper-step step="4"> Configuration <small>Select Configuration</small> </v-stepper-step>
          <v-stepper-content step="4" class="mt-2"></v-stepper-content>
        </v-stepper>
        <div style="width: 75%; height: 500px" class="px-10">
          <ProjectInfoSetting v-if="step === 1" @next="next" />
          <DatasetSelector v-else-if="step === 2" @next="next" @prev="prev" @skip="skip" />
          <TargetSelector v-else-if="step === 3" @next="next" @prev="prev" />
          <ProjectConfigurationSetting v-else @prev="prev" @create="create" />
        </div>
      </div>
    </v-card>
  </v-dialog>
</template>
<script>
import Swal from "sweetalert2";

import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProjectInfoSetting from "./stepper/ProjectInfoSetting.vue";
import DatasetSelector from "./stepper/DatasetSelector.vue";
import TargetSelector from "./stepper/TargetSelector.vue";
import ProjectConfigurationSetting from "./stepper/ProjectConfigurationSetting.vue";

import { TaskType, LearningType, ContainerName } from "@/shared/enums";

import { updateProjectInfo, setWorkflow } from "@/api";

export default {
  components: { ProjectInfoSetting, DatasetSelector, TargetSelector, ProjectConfigurationSetting },

  props: {
    step: {
      default: 1
    }
  },

  data() {
    return {
      TaskType,
      LearningType,
      dialog: false
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"])
  },

  mounted() {
    this.$EventBus.$on("forcedTermination", () => {
      this.dialog = false;
      this.$emit("stepChange", 1);
      this.$emit("close");
    });
  },

  beforeDestroy() {
    this.INIT_PROJECT();
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_PROJECT: ProjectMutations.SET_PROJECT,
      INIT_PROJECT: ProjectMutations.INIT_PROJECT
    }),

    async skip() {
      const result = await Swal.fire({
        title: `SKIP`,
        html: "<div>건너뛰기 시 Task Type이 CHAT으로 고정됩니다.<br/>그래도 SKIP하시겠습니까?</div>",
        icon: "info",
        showCancelButton: true,
        allowOutsideClick: false,
        allowEscapeKey: false,
        confirmButtonColor: "#3085d6",
        cancelButtonColor: "#d33",
        confirmButtonText: "확인",
        cancelButtonText: "취소"
      });

      console.log("result", result);

      if (result.isConfirmed && this.step !== 4) {
        this.$emit("stepChange", this.step + 1);
      }
    },

    async next(data) {
      if (this.step !== 4) {
        this.$emit("stepChange", this.step + 1);
      }
      this.SET_PROJECT(data);

      if (this.project.id) {
        const param = {
          project_id: this.project.id,
          project_target: this.project.target_id || "",
          project_dataset: this.project.dataset || "",
          task_type: this.project.task_type || "",
          learning_type: this.project.learning_type || "",
          weight_file: this.project.weight_file || "",
          autonn_dataset_file: this.project.autonn_dataset_file || "",
          autonn_base_model: this.project.autonn_basemodel || "",
          nas_type: this.project.nas_type || "",
          deploy_weight_level: this.project.deploy_weight_level || "",
          deploy_precision_level: this.project.deploy_precision_level || "",
          deploy_processing_lib: this.project.deploy_processing_lib || "",
          deploy_user_edit: this.project.deploy_user_edit || "no",
          deploy_input_method: this.project.deploy_input_method || "",
          deploy_input_data_path: this.project.deploy_input_data_path || "",
          deploy_output_method: this.project.deploy_output_method || "0",

          deploy_input_source: this.project.deploy_input_source || "0"
        };

        await updateProjectInfo(param);
      }
    },

    prev() {
      if (this.step !== 1) {
        this.$emit("stepChange", this.step - 1);
      }
    },

    async create(data) {
      this.SET_PROJECT(data);

      const param = {
        project_id: this.project.id,
        project_target: this.project.target_id,
        project_dataset: this.project.dataset || "",
        task_type: this.project.task_type,
        learning_type: this.project.learning_type || "",
        weight_file: this.project.weight_file || "",
        autonn_dataset_file: this.project.autonn_dataset_file,
        autonn_base_model: this.project.autonn_basemodel,
        nas_type: this.project.nas_type,
        deploy_weight_level: this.project.deploy_weight_level,
        deploy_precision_level: this.project.deploy_precision_level,
        deploy_processing_lib: this.project.deploy_processing_lib,
        deploy_user_edit: this.project.deploy_user_edit || "no",
        deploy_input_method: this.project.deploy_input_method,
        deploy_input_data_path: this.project.deploy_input_data_path,
        deploy_output_method: this.project.deploy_output_method || "0",

        deploy_input_source: this.project.deploy_input_source || "0"
      };

      await updateProjectInfo(param);

      const workflow = [ContainerName.AUTO_NN, ContainerName.CODE_GEN, ContainerName.IMAGE_DEPLOY];

      if (this.project.deploy_user_edit === "yes") {
        workflow.splice(workflow.length - 1, 0, ContainerName.USER_EDITING);
      }

      await setWorkflow(this.project.id, workflow);
      
      // 프로젝트 생성 완료 이벤트 발생
      this.$EventBus.$emit("projectCreated");
      
      // 다이얼로그 닫기
      this.close();
      
        // 프로젝트 ID 안전하게 보존
        const createdProjectId = this.project.id;
        console.log("일반 프로젝트 ID 보존:", createdProjectId);
        
        // 약간의 지연 후 페이지 이동 (목록 업데이트 완료 후)
        setTimeout(() => {
          if (this.$route.params?.id) {
            this.$router.go();
          } else {
            if (createdProjectId) {
              this.$router.push(`/project/${createdProjectId}`);
            } else {
              console.error("일반 프로젝트 ID가 없어 페이지 이동 불가");
            }
          }
        }, 200);
    },

    close() {
      this.dialog = false;
      this.$emit("stepChange", 1);
      this.$emit("close");
      // this.INIT_PROJECT();
    }
  }
};
</script>
<style lang="scss" scoped></style>
