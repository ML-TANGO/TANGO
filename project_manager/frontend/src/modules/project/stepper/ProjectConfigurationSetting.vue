<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <p class="text-h5 mb-4" style="color: #4a80ff">Configuration</p>

      <!-- TASK TYPE -->
      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">Task Type</div>
        <v-radio-group v-model="taskType" row hide-details="" class="ma-0" :disabled="isEmpty(this.selectedImage)">
          <v-radio label="Classification" :value="TaskType.CLASSIFICATION"></v-radio>
          <v-radio label="Detection" :value="TaskType.DETECTION"></v-radio>
          <!-- <v-radio label="Chat" :value="TaskType.CHAT"></v-radio> -->
        </v-radio-group>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">Learning Type</div>
        <v-radio-group v-model="learningType" row hide-details="" class="ma-0" :disabled="isEmpty(this.selectedImage)">
          <v-radio label="Normal" :value="LearningType.NORMAL"></v-radio>
          <v-radio label="Incremental" :value="LearningType.INCREMENTAL" v-if="isIncremental"></v-radio>
          <v-radio label="Transfer" :value="LearningType.TRANSFER"></v-radio>
          <v-radio label="HPO" :value="LearningType.HPO"></v-radio>
        </v-radio-group>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <div v-if="learningType === LearningType.TRANSFER">
        <div class="d-flex align-center" style="gap: 25px">
          <div style="width: 150px">Weight File</div>
          <v-text-field
            :value="weightFilePath"
            :key="`weight file`"
            outlined
            dense
            :label="'Weight File Selector'"
            hide-details
            readonly
            @click="onClickWeightFilePath"
          />
          <!-- 
            :label="weightFilePath || 'Weight File Selector'"

           -->
        </div>

        <v-divider class="mt-3 mb-3"></v-divider>
      </div>

      <!-- AUTONN -->
      <!-- <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">AutonNN Config</div>
        <div class="d-flex align-center" style="gap: 5px">
          <v-text-field v-model="dataset" outlined dense label="Dataset file" hide-details readonly />
        </div>

        <div class="d-flex align-center" style="gap: 5px">
          <v-text-field v-model="basemodel" outlined dense label="Base Model" hide-details readonly />
        </div>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider> -->

      <!-- nas Type -->
      <!-- <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">NAS Type</div>
        <v-radio-group v-model="nasType" row hide-details class="ma-0">
          <v-radio label="Backbone Nas" value="bb_nas"></v-radio>
          <v-radio label="Neck Nas" value="neck_nas"></v-radio>
        </v-radio-group>
      </div>

      <v-divider class="mt-3 mb-3" v-if="selectedTarget.info !== 'ondevice'"></v-divider> -->

      <!-- Input Source -->
      <div class="d-flex" style="gap: 25px">
        <div style="min-width: 150px">Input Source</div>
        <div class="d-flex flex-column" style="gap: 25px">
          <div class="d-flex align-center" style="gap: 15px">
            <div style="width: 40%">
              <v-combobox dense hide-details outlined :items="inputSourceItems" v-model="inputSourceType"></v-combobox>
            </div>
            <div style="width: 60%">
              <v-text-field
                :key="`inputsource-${inputSourceKey}`"
                v-if="inputSourceType === 'Camera ID'"
                :value="inputSource"
                type="number"
                outlined
                dense
                label="Camera ID (0~9)"
                hide-details
                @change="inputSourceChange"
              />
              <v-text-field
                v-else
                v-model="inputSource"
                outlined
                dense
                :label="inputSourceLabel(inputSourceType)"
                hide-details
              />
            </div>
          </div>
        </div>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <!-- outPut Method -->
      <div class="d-flex" style="gap: 25px">
        <div style="min-width: 150px">Output Method</div>
        <div class="d-flex flex-column" style="gap: 25px">
          <div class="d-flex align-center" style="gap: 15px">
            <div style="width: 40%">
              <v-autocomplete
                dense
                hide-details
                outlined
                :items="outputMethodItems"
                v-model="outputMethodType"
              ></v-autocomplete>
            </div>
            <div style="width: 60%">
              <v-text-field
                :disabled="outputMethodType !== 'URL or Directory Path'"
                v-model="outputMethod"
                outlined
                dense
                :label="outputSourceLabel(outputMethodType)"
                hide-details
              />
            </div>
          </div>
        </div>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <!-- Deploy Config -->
      <div class="d-flex" style="gap: 25px">
        <div style="min-width: 150px">Deploy Config</div>
        <div class="d-flex flex-column" style="gap: 25px">
          <div class="d-flex" style="gap: 10px">
            <v-tooltip top :key="`lightWeightLv-${lightWeightLvKey}`">
              <template v-slot:activator="{ on, attrs }">
                <v-text-field
                  v-bind="attrs"
                  v-on="on"
                  :value="lightWeightLv"
                  type="number"
                  outlined
                  dense
                  label="Light Weight Level"
                  hide-details
                  @change="lightWeightLvChange"
                />
              </template>
              <span style="font-size: 13px">
                0 .. 10, that specifies the level of model optimization, 10 means "maximal"
              </span>
            </v-tooltip>

            <v-tooltip top :key="`precisionLv-${precisionLvKey}`">
              <template v-slot:activator="{ on, attrs }">
                <v-text-field
                  v-bind="attrs"
                  v-on="on"
                  :value="precisionLv"
                  type="number"
                  outlined
                  dense
                  label="Precision Level"
                  hide-details
                  @change="precisionLvChange"
                />
              </template>
              <span style="font-size: 13px">
                0 .. 10, that specifies the level of precision, 10 means "do not modify neural"
              </span>
            </v-tooltip>

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

            <!-- <v-text-field v-model="processingLib" outlined dense la
            bel="Processing Lib" hide-details readonly /> -->
          </div>

          <!-- <div class="d-flex" style="gap: 10px">
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
          </div> -->
          <!-- <div class="d-flex" style="gap: 10px">
            <v-text-field v-model="inputPath" outlined dense label="Input Data Path" hide-details style="width: 50%" />
          </div> -->
        </div>
      </div>
    </div>
    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="pre"> PREV </v-btn>
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="create"> FINISH </v-btn>
    </div>

    <!-- <v-dialog v-model="openWeightFileDialog"> </v-dialog> -->
    <PTFileSelector ref="PTFileSelector" :structure="structure" @select="onSelectPtFile" />
  </div>
</template>
<script>
import { mapState } from "vuex";
import { ProjectNamespace } from "@/store/modules/project";
import { LearningType, TaskType, CommonDatasetName } from "@/shared/enums";

import PTFileSelector from "@/modules/common/dialog/PTFileSelector.vue";

import { get_common_folder_structure } from "@/api";
export default {
  components: { PTFileSelector },

  data() {
    return {
      taskType: "",
      learningType: "",
      weightFilePath: "",
      openWeightFileDialog: false,
      nasType: "",
      basemodel: "basemode.yaml",
      dataset: "dataset.yaml",
      lightWeightLv: 5,
      precisionLv: 5,
      processingLib: "cv2",
      userEditing: "no",
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
      // outputMethod: "",
      // outputMethodItem: [
      //   { value: "console", label: "Console" },
      //   { value: "graphic", label: "Graphic" },
      //   { value: "mp4", label: "MP4" }
      // ],

      // =================================
      inputSourceItems: ["Camera ID", "URL or File/Directory Path"],
      inputSourceType: "Camera ID",
      inputSource: null,

      outputMethodItems: ["Screen Display", "Text Output", "URL or Directory Path"],
      outputMethodType: "Screen Display",
      outputMethod: null,

      lightWeightLvKey: 0,
      precisionLvKey: 0,
      inputSourceKey: 0,

      structure: [],

      LearningType,
      TaskType
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["selectedImage", "selectedTarget", "project"]),

    isIncremental() {
      return this.selectedImage?.name === CommonDatasetName.COCO && this.taskType === TaskType.DETECTION;
    }
  },

  watch: {
    outputMethodType() {
      if (this.outputMethodType === "Screen Display") {
        this.outputMethod = "0";
      } else if (this.outputMethodType === "Text Output") {
        this.outputMethod = "1";
      } else {
        this.outputMethod = this.project.deploy_output_method || "0";
      }
    },

    isIncremental() {
      if (!this.isIncremental && this.learningType === LearningType.INCREMENTAL) {
        this.learningType = LearningType.NORMAL;
      }
    }
  },

  mounted() {
    this.taskType = this.project.task_type || TaskType.DETECTION;
    this.learningType = this.project.learning_type || LearningType.NORMAL;
    this.weightFilePath = this.project?.weight_file || "";
    this.nasType = this.project.nas_type || "neck_nas";
    this.basemodel = this.project.autonn_basemodel || "basemode.yaml";
    this.dataset = this.project.autonn_dataset_file || "dataset.yaml";
    this.lightWeightLv = this.project.deploy_weight_level || 5;
    this.precisionLv = this.project.deploy_precision_level || 5;
    this.processingLib = this.project.deploy_processing_lib || "cv2";
    this.userEditing = this.project.deploy_user_edit || "no";
    this.inputMethod = this.project.deploy_input_method;
    this.inputPath = this.project.deploy_input_data_path;
    this.outputMethod = this.project.deploy_output_method || "0";

    // new
    this.inputSource = this.project.deploy_input_source || "0";

    if (this.outputMethod.toString() === "0") {
      this.outputMethodType = "Screen Display";
    } else if (this.outputMethod.toString() === "1") {
      this.outputMethodType = "Text Output";
    } else {
      this.outputMethodType = "URL or Directory Path";
    }

    if (!isNaN(this.inputSource) && Number(this.inputSource) >= 0 && Number(this.inputSource) <= 9) {
      this.inputSourceType = "Camera ID";
    } else {
      this.inputSourceType = "URL or File/Directory Path";
    }

    if (!this.isIncremental && this.learningType === LearningType.INCREMENTAL) {
      // 선택 한 데이터 셋이 COCO가 아니고, INCREMENTAL이 선택되어있을 경우 Normal로 강제로 변경
      this.learningType = LearningType.NORMAL;
    }

    if (this.isEmpty(this.selectedImage)) {
      this.taskType = TaskType.CHAT;
      this.learningType = LearningType.NORMAL;
    }
  },

  methods: {
    isEmpty(obj) {
      if (!obj) return true;
      return Object.keys(obj).length === 0 && obj.constructor === Object;
    },

    pre() {
      this.$emit("prev");
    },
    create() {
      this.$emit("create", {
        task_type: this.taskType,
        learning_type: this.learningType,
        weight_file: this.weightFilePath || "",
        nas_type: this.nasType,
        autonn_basemodel: this.basemodel || "",
        autonn_dataset_file: this.dataset || "",
        deploy_input_data_path: this.inputPath || "",
        deploy_output_method: this.outputMethod || "",
        deploy_precision_level: this.precisionLv || "",
        deploy_processing_lib: this.processingLib || "",
        deploy_user_edit: this.userEditing || "no",
        deploy_weight_level: this.lightWeightLv || "",
        deploy_input_method: this.inputMethod || "0",

        deploy_input_source: this.inputSource || "0"
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
    },
    lightWeightLvChange(value) {
      this.lightWeightLv = Number(value) > 10 ? 10 : Number(value) < 0 ? 0 : Number(value);
      this.lightWeightLvKey++;
    },
    precisionLvChange(value) {
      this.precisionLv = Number(value) > 10 ? 10 : Number(value) < 0 ? 0 : Number(value);
      this.precisionLvKey++;
    },
    inputSourceChange(value) {
      if (this.inputSourceType === "Camera ID")
        this.inputSource = Number(value) > 9 ? 9 : Number(value) < 0 ? 0 : Number(value);
      else this.inputSource = value;
      this.inputSourceKey++;
    },
    inputSourceLabel(inputSource) {
      if (inputSource.toLowerCase() === "url") {
        return "URL to receive input image stream";
      } else {
        return inputSource;
      }
    },
    outputSourceLabel(inputSource) {
      if (inputSource.toLowerCase() === "url") {
        return "URL to send output image stream";
      } else {
        return inputSource;
      }
    },

    async onClickWeightFilePath() {
      // this.openWeightFileDialog = true;
      const res = await get_common_folder_structure();
      this.structure = res.structure;

      const ptFileSelector = this.$refs.PTFileSelector;
      ptFileSelector.isOpen = true;
    },

    onSelectPtFile(path) {
      console.log("onSelectPtFile - path", path);
      this.weightFilePath = path;
    }
  }
};
</script>
<style lang="scss" scoped></style>
