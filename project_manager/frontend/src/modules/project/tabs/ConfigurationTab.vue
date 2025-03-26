<template>
  <div class="d-flex flex-column" style="gap: 15px; margin-left: -16px">
    <div>
      <h4 class="ml-3 mb-3">Description</h4>
      <v-text-field v-model="description" class="mx-3" dense outlined hide-details />
    </div>

    <div class="d-flex align-center">
      <div style="width: 25%">
        <h4 class="ml-3 mt-3">Task Type</h4>
        <v-radio-group v-model="taskType" row hide-details class="ma-0 mt-2 ml-3" readonly>
          <v-radio label="Classification" :value="TaskType.CLASSIFICATION"></v-radio>
          <v-radio label="Detection" :value="TaskType.DETECTION"></v-radio>
          <!-- <v-radio label="Chat" :value="TaskType.CHAT"></v-radio> -->
        </v-radio-group>
      </div>
    </div>

    <div class="d-flex">
      <div style="width: 50%" v-if="selectedImage">
        <!-- <div style="width: 50%"> -->
        <h4 class="ml-3 mt-3">Dataset</h4>
        <DatasetCard v-if="isDatasetLoading" :item="selectedImage" />
        <SkeletonLoaderCard v-else />
      </div>
      <div :style="{ width: selectedImage ? '50%' : '100%' }">
        <!-- <div style="width: 50%"> -->
        <h4 class="ml-3 mt-3">Target</h4>
        <TargetCard v-if="isTargetLoading" ref="targetCaredRef" :item="selectedTarget" />
        <SkeletonLoaderCard v-else />
      </div>
    </div>

    <h4 class="ml-3 mt-3" style="min-width: 150px">Deploy Config</h4>
    <div class="d-flex flex-column ml-3 mt-3" style="gap: 25px">
      <div class="d-flex" style="gap: 10px">
        <v-text-field
          v-model="lightWeightLv"
          type="number"
          outlined
          dense
          label="Light Weight Level"
          hide-details
          readonly
        />
        <v-text-field
          v-model="precisionLv"
          type="number"
          outlined
          dense
          label="Precision Level"
          hide-details
          readonly
        />
        <v-autocomplete
          v-model="userEditing"
          :items="userEditingItem"
          label="User Editing"
          outlined
          dense
          hide-details
          item-text="label"
          readonly
        />
      </div>

      <div class="d-flex" style="gap: 10px">
        <v-text-field
          v-model="inputSource"
          label="Input Source"
          outlined
          dense
          hide-details
          item-text="label"
          readonly
        />

        <v-text-field
          v-model="outputMethod"
          label="Output Method"
          outlined
          dense
          hide-details
          item-text="label"
          readonly
        />
      </div>
    </div>
  </div>
</template>
<script>
// import Vue from "vue";

import DatasetCard from "@/modules/common/card/DatasetCard.vue";
import TargetCard from "@/modules/common/card/TargetCardV2.vue";
import SkeletonLoaderCard from "@/modules/common/card/SkeletonLoaderCard.vue";

import { TaskType } from "@/shared/enums";

// import { getTargetInfo, getDatasetListTango, getDatasetFolderSize, getDatasetFileCount } from "@/api";
export default {
  components: { DatasetCard, TargetCard, SkeletonLoaderCard },

  props: {
    projectInfo: {
      default: null
    },

    isOpen: {
      default: false
    }
  },

  data() {
    return {
      TaskType,
      taskType: "",
      nasType: "",
      description: "",
      baseModel: "basemode.yaml",
      datasetFile: "dataset.yaml",
      lightWeightLv: 0,
      precisionLv: 0,
      processingLib: "cv2",
      userEditing: "",
      inputSource: "",
      userEditingItem: [
        { value: "yes", label: "Yes" },
        { value: "no", label: "No" }
      ],
      inputMethod: "",
      inputMethodItem: [
        { value: "camera", label: "Camera" },
        { value: "mp4", label: "MP4" },
        { value: "picture", label: "Picture" },
        { value: "folder", label: "Folder" }
      ],
      inputPath: "/data",
      outputMethod: "",
      outputMethodItem: [
        { value: "console", label: "Console" },
        { value: "graphic", label: "Graphic" },
        { value: "mp4", label: "MP4" }
      ],

      selectedTarget: null,
      isTargetLoading: false,

      selectedImage: null,
      isDatasetLoading: true
    };
  },

  watch: {
    async projectInfo() {
      if (this.projectInfo) {
        this.taskType = this.projectInfo.task_type;
        this.nasType = this.projectInfo.nas_type;
        this.description = this.projectInfo.project_description;
        this.baseModel = this.projectInfo.autonn_basemodel || "basemode.yaml";
        this.datasetFile = this.projectInfo.autonn_dataset_file || "dataset.yaml";
        this.lightWeightLv = this.projectInfo.deploy_weight_level;
        this.precisionLv = this.projectInfo.deploy_precision_level;
        this.processingLib = this.projectInfo.deploy_processing_lib || "cv2";
        this.userEditing = this.projectInfo.deploy_user_edit;
        this.inputMethod = this.projectInfo.deploy_input_method;
        this.inputPath = this.projectInfo.deploy_input_data_path;
        this.outputMethod = this.projectInfo.deploy_output_method;
        this.inputSource = this.projectInfo.deploy_input_source;

        this.selectedTarget = this.projectInfo?.target_info;
        this.isTargetLoading = true;
        this.selectedImage = this.projectInfo?.datasetObject;
        this.isDatasetLoading = true;
      }
    },

    isOpen: {
      immediate: true,
      deep: true,
      handler() {
        this.$nextTick(() => {
          if (this.isOpen) {
            const targetCaredRef = this.$refs.targetCaredRef;
            if (!targetCaredRef) return;
            targetCaredRef.setAnimation();
          }
        });
      }
    }
  },

  methods: {}
};
</script>
<style lang="scss" scoped></style>
