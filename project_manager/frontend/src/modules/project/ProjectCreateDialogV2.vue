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

import { ContainerName, TaskType, LearningType } from "@/shared/enums";

import { updateProjectInfo, setWorkflow, createProject } from "@/api";

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

    async next(data) {
      this.SET_PROJECT(data);

      // Segmentation + Continual Learning 선택 시 단계 스킵 로직
      if (
        this.step === 2 &&
        this.project.task_type === TaskType.SEGMENTATION &&
        this.project.learning_type === LearningType.CONTINUAL_LEARNING
      ) {
        // Dataset, Target, Hyperparameter, Option 단계를 스킵하고 바로 프로젝트 생성
        await this.createSegmentationProject();
        return;
      }

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

    async createSegmentationProject() {
      /**
       * Segmentation + Continual Learning 프로젝트 생성
       * Dataset, Target, Hyperparameter, Option 단계를 스킵하고 기본값으로 프로젝트 생성
       */
      try {
        console.log("Segmentation 프로젝트 생성 시작");
        console.log("- 현재 project 객체:", this.project);
        
        // 1단계: 프로젝트 기본 정보가 없으면 먼저 생성
        let projectData = { ...this.project };
        
        if (!projectData.id) {
          console.log("프로젝트 기본 정보 생성 필요");
          const projectCreationResult = await createProject(
            projectData.project_name, 
            projectData.project_description || ""
          );
          console.log("프로젝트 기본 정보 생성 완료:", projectCreationResult);
          
          // 생성된 프로젝트 정보로 업데이트
          projectData = {
            ...projectData,
            id: projectCreationResult.id,
            project_name: projectCreationResult.name,
            project_description: projectCreationResult.description
          };
          
          // Vuex store도 업데이트
          this.SET_PROJECT(projectData);
        }
        
        // 2단계: Segmentation 전용 설정 추가
        projectData = {
          ...projectData,
          task_type: TaskType.SEGMENTATION,
          learning_type: LearningType.CONTINUAL_LEARNING,
          // 기본값들로 설정 (필요시 조정 가능)
          target_id: null, // 추후 AutoNN_CL에서 처리
          dataset: "", // Segmentation은 dataset 단계 스킵
          autonn_dataset_file: "dataset.yaml", // 기본 설정
          autonn_basemodel: "basemodel.yaml", // 기본 설정
          nas_type: "nas" // 기본 설정
        };
        
        console.log("📋 최종 프로젝트 데이터:", projectData);

        const param = {
          project_id: projectData.id,
          project_target: projectData.target_id || "",
          project_dataset: projectData.dataset || "",
          task_type: projectData.task_type,
          learning_type: projectData.learning_type,
          weight_file: projectData.weight_file || "",
          autonn_dataset_file: projectData.autonn_dataset_file,
          autonn_base_model: projectData.autonn_basemodel,
          nas_type: projectData.nas_type,
          // Segmentation 프로젝트에 필수인 배포 설정값들 - Configuration에서 입력받은 값 또는 기본값
          deploy_weight_level: projectData.deploy_weight_level || "5",
          deploy_precision_level: projectData.deploy_precision_level || "5",
          deploy_processing_lib: projectData.deploy_processing_lib || "cv2",
          deploy_user_edit: projectData.deploy_user_edit || "no",
          deploy_input_method: projectData.deploy_input_method || "0",
          deploy_input_data_path: projectData.deploy_input_data_path || "",
          deploy_output_method: projectData.deploy_output_method || "0",
          deploy_input_source: projectData.deploy_input_source || "0"
        };

        console.log("updateProjectInfo 호출...");
        await updateProjectInfo(param);
        console.log("updateProjectInfo 완료");

        // Segmentation 전용 워크플로우 설정 (추후 AutoNN_CL 컨테이너 추가)
        const workflow = ["autonn_cl"]; // AutoNN_CL 컨테이너만 사용
        console.log("워크플로우 설정:", workflow, "for project ID:", projectData.id);
        await setWorkflow(projectData.id, workflow);
        console.log("워크플로우 설정 완료");
        
        // 프로젝트 ID를 안전하게 보존
        const createdProjectId = projectData.id;
        console.log("생성된 프로젝트 ID 보존:", createdProjectId);

        // 프로젝트 생성 완료 이벤트 발생
        this.$EventBus.$emit("projectCreated");
        
        // 다이얼로그 닫기
        this.close();
        
        // 약간의 지연 후 페이지 이동 (목록 업데이트 완료 후)
        setTimeout(() => {
          console.log("페이지 이동 준비:");
          console.log("- createdProjectId:", createdProjectId);
          console.log("- this.project.id:", this.project.id);
          console.log("- $route.params:", this.$route.params);
          
          if (this.$route.params?.id) {
            console.log("기존 프로젝트 수정 - 페이지 새로고침");
            this.$router.go();
          } else {
            // 생성된 프로젝트 ID 사용
            if (createdProjectId) {
              console.log(`🚀 새 프로젝트 생성 - /project/${createdProjectId}로 이동`);
              this.$router.push(`/project/${createdProjectId}`);
            } else {
              console.error("프로젝트 ID가 없어 페이지 이동 불가");
              console.error("- createdProjectId:", createdProjectId);
              console.error("- this.project:", this.project);
            }
          }
        }, 200);
      } catch (error) {
        console.error("Segmentation 프로젝트 생성 실패:", error);
        // 에러 처리 로직 추가 (추후 토스트 메시지 등으로 사용자에게 알림)
      }
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