<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <p class="text-h5 mb-4" style="color: #4a80ff">Configuration</p>

      <!-- TASK TYPE -->
      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">TASK TYPE</div>
        <v-radio-group v-model="taskType" row hide-details="" class="ma-0" readonly>
          <v-radio label="Classification" value="classification"></v-radio>
          <v-radio label="Detection" value="detection"></v-radio>
        </v-radio-group>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <!-- AUTONN -->
      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">AutonNN Config</div>
        <div class="d-flex align-center" style="gap: 5px">
          <v-text-field v-model="dataset" outlined dense label="Dataset file" hide-details readonly />
        </div>

        <div class="d-flex align-center" style="gap: 5px">
          <v-text-field v-model="basemodel" outlined dense label="Base Model" hide-details readonly />
        </div>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <!-- nas Type -->
      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">NAS Type</div>
        <v-radio-group v-model="nasType" row hide-details class="ma-0">
          <v-radio label="Backbone Nas" value="bb_nas"></v-radio>
          <v-radio label="Neck Nas" value="neck_nas"></v-radio>
        </v-radio-group>
      </div>

      <v-divider class="mt-3 mb-3" v-if="selectedTarget.info !== 'ondevice'"></v-divider>

      <!-- Deploy Config -->
      <div class="d-flex" style="gap: 25px" v-if="selectedTarget.info !== 'ondevice'">
        <div style="min-width: 150px">Deploy Config</div>
        <div class="d-flex flex-column" style="gap: 25px">
          <div class="d-flex" style="gap: 10px">
            <v-text-field
              v-model="lightWeightLv"
              type="number"
              outlined
              dense
              label="Light Weight Level"
              hide-details
            />
            <v-text-field v-model="precisionLv" type="number" outlined dense label="Precision Level" hide-details />
            <v-text-field v-model="processingLib" outlined dense label="Processing Lib" hide-details readonly />
          </div>

          <div class="d-flex" style="gap: 10px">
            <v-autocomplete
              :value="inputMethod"
              :items="inputMethodItem"
              label="Input Method"
              outlined
              dense
              hide-details
              item-text="label"
              @change="inputMethodChange"
              style="width: 50%"
            />
            <v-text-field v-model="inputPath" outlined dense label="Input Data Path" hide-details style="width: 50%" />
          </div>
          <div class="d-flex" style="gap: 10px">
            <v-autocomplete
              :value="outputMethod"
              :items="outputMethodItem"
              label="Output Method"
              outlined
              dense
              hide-details
              item-text="label"
              @change="outputMethodChange"
            />

            <v-autocomplete
              :value="userEditing"
              :items="userEditingItem"
              label="User Editing"
              outlined
              dense
              hide-details
              item-text="label"
              @change="userEditingChange"
            />
          </div>
        </div>
      </div>
    </div>
    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="pre"> PREV </v-btn>
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="create"> FINISH </v-btn>
    </div>
  </div>
</template>
<script>
import { mapState } from "vuex";
import { ProjectNamespace } from "@/store/modules/project";
export default {
  data() {
    return {
      taskType: "",
      nasType: "",
      basemodel: "basemode.yaml",
      dataset: "dataset.yaml",
      lightWeightLv: 0,
      precisionLv: 0,
      processingLib: "cv2",
      userEditing: "",
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
      inputPath: "",
      outputMethod: "",
      outputMethodItem: [
        { value: "console", label: "Console" },
        { value: "graphic", label: "Graphic" },
        { value: "mp4", label: "MP4" }
      ]
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["selectedImage", "selectedTarget", "project"])
  },

  created() {
    this.taskType = this.selectedImage.OBJECT_TYPE === "C" ? "classification" : "detection";
    this.nasType = this.project.nas_type;
    this.basemodel = this.project.autonn_basemodel || "basemode.yaml";
    this.dataset = this.project.autonn_dataset_file || "dataset.yaml";
    this.lightWeightLv = this.project.deploy_weight_level;
    this.precisionLv = this.project.deploy_precision_level;
    this.processingLib = this.project.deploy_processing_lib || "cv2";
    this.userEditing = this.project.deploy_user_edit;
    this.inputMethod = this.project.deploy_input_method;
    this.inputPath = this.project.deploy_input_data_path;
    this.outputMethod = this.project.deploy_output_method;
  },

  methods: {
    pre() {
      this.$emit("prev");
    },
    create() {
      this.$emit("create", {
        task_type: this.taskType,
        nas_type: this.nasType,
        autonn_basemodel: this.basemodel || "",
        autonn_dataset_file: this.dataset || "",
        deploy_input_data_path: this.inputPath || "",
        deploy_output_method: this.outputMethod || "",
        deploy_precision_level: this.precisionLv || "",
        deploy_processing_lib: this.processingLib || "",
        deploy_user_edit: this.userEditing || "",
        deploy_weight_level: this.lightWeightLv || "",
        deploy_input_method: this.inputMethod || ""
      });
    },

    inputMethodChange(value) {
      this.inputMethod = value;
    },
    outputMethodChange(value) {
      this.outputMethod = value;
    },
    userEditingChange(value) {
      this.userEditing = value;
    }
  }
};
</script>
<style lang="scss" scoped></style>
