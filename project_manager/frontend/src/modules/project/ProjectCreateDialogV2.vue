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
          <v-stepper-step :complete="step > 1" step="1">
            Project Info <small class="mt-2">Enter Project Info</small>
          </v-stepper-step>
          <!-- <v-stepper-content step="1" class="my-1"> </v-stepper-content> -->

          <v-stepper-step :complete="step > 2" step="2"
            >Configuration <small class="mt-2">Select Configuration</small>
          </v-stepper-step>
          <!-- <v-stepper-content step="2" class="my-3"></v-stepper-content> -->

          <v-stepper-step :complete="step > 3" step="3">
            Dataset <small class="mt-2">Select Dataset</small>
          </v-stepper-step>
          <!-- <v-stepper-content step="3" class="my-3"></v-stepper-content> -->

          <v-stepper-step :complete="step > 4" step="4">
            Target <small class="mt-2">Select Target</small>
          </v-stepper-step>
          <!-- <v-stepper-content step="4" class="my-3"></v-stepper-content> -->

          <v-stepper-step :complete="step > 5" step="5">
            Hyperparameter <small class="mt-2">Edit Hyperparameter YAML</small>
          </v-stepper-step>
          <!-- <v-stepper-content step="5" class="my-3"></v-stepper-content> -->

          <v-stepper-step :complete="step > 6" step="6">
            Option <small class="mt-2">Edit Option YAML</small>
          </v-stepper-step>
          <!-- <v-stepper-content step="6" class="my-3"></v-stepper-content> -->
        </v-stepper>
        <div style="width: 75%; height: 500px" class="px-10">
          <ProjectInfoSetting v-if="step === 1" @next="next" />
          <ProjectConfigurationSetting v-else-if="step === 2" @next="next" @prev="prev" />
          <DatasetSelector v-else-if="step === 3" @next="next" @prev="prev" @skip="skip" />
          <TargetSelector v-else-if="step === 4" @next="next" @prev="prev" />
          <EditHyperparameterFile v-else-if="step === 5" :project="project" @next="next" @prev="prev" />
          <EditArgumentsFile v-else :project="project" @create="create" @prev="prev" />
        </div>
      </div>
    </v-card>
  </v-dialog>
</template>
<script>
import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import ProjectInfoSetting from "./stepper/ProjectInfoSetting.vue";
import DatasetSelector from "./stepper/DatasetSelector.vue";
import TargetSelector from "./stepper/TargetSelector.vue";
import EditHyperparameterFile from "./stepper/EditHyperparameterFile.vue";
import EditArgumentsFile from "./stepper/EditArgumentsFile.vue";
import ProjectConfigurationSetting from "./stepper/ProjectConfigurationSettingV2.vue";

import { ContainerName, TaskType } from "@/shared/enums";

import { updateProjectInfo, setWorkflow } from "@/api";

export default {
  components: {
    ProjectInfoSetting,
    DatasetSelector,
    TargetSelector,
    ProjectConfigurationSetting,
    EditHyperparameterFile,
    EditArgumentsFile
  },

  props: {
    step: {
      default: 1
    }
  },

  data() {
    return {
      TaskType,
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

    async next(data) {
      this.SET_PROJECT(data);

      if (this.step !== 6) {
        const nextStep = 1;
        // if (this.step === 2 && this.project.task_type === TaskType.CHAT) nextStep++;
        this.$emit("stepChange", this.step + nextStep);
      }

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
        const prevStep = 1;
        // if (this.step === 4 && this.project.task_type === TaskType.CHAT) prevStep++;
        this.$emit("stepChange", this.step - prevStep);
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
      if (this.$route.params?.id) {
        this.$router.go();
      } else {
        this.$router.push(`project/${this.project.id}`);
      }
      this.close();
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
