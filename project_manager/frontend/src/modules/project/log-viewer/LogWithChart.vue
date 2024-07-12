<template>
  <div style="padding: 8px; overflow-x: auto">
    <!-- <div v-for="([key, value], index) in Object.entries(displayData)" :key="index" class="py-2">
      <div class="mb-1">update_id : {{ key }}</div>
      <div style="white-space: normal">{{ value }}</div>
      <v-divider></v-divider>
    </div> -->
    <div class="d-flex flex-wrap" style="gap: 38px">
      <div class="hyper">
        <AutonnSetting
          :data="displayData['hyperparameter']"
          :isCompleted="displayData['progress'] >= AutonnStatus.PROJECT_INFO"
        />
      </div>
      <AutonnSystem :data="displayData['system']" :isCompleted="displayData['progress'] >= AutonnStatus.SYSTEM" />
      <AutonnModel :data="displayData['model_summary']" :isCompleted="displayData['progress'] >= AutonnStatus.MODEL" />
      <AutonnDataset
        :valDataset="displayData['val_dataset']"
        :trainDataset="displayData['train_dataset']"
        :isCompleted="displayData['progress'] >= AutonnStatus.DATASET"
      />
    </div>
    <div style="width: 100%" class="mt-8">
      <AutonnTrain
        :train="displayData['train_loss_latest']"
        :val="displayData['val_accuracy_latest']"
        :trainLastSteps="displayData['train_loss_laststep_list']"
        :valLastSteps="displayData['val_accuracy_laststep_list']"
        :epochSummary="displayData['epoch_summary_list']"
      />
    </div>
  </div>
</template>
<script>
import AutonnSetting from "./steps/AutonnSetting.vue";
import AutonnSystem from "./steps/AutonnSystem.vue";
import AutonnModel from "./steps/AutonnModel.vue";
import AutonnDataset from "./steps/AutonnDataset.vue";
import AutonnTrain from "./steps/AutonnTrain.vue";

import { AutonnStatus } from "@/shared/enums";
export default {
  components: { AutonnSetting, AutonnSystem, AutonnModel, AutonnDataset, AutonnTrain },

  props: {
    data: {
      default: null
    }
  },

  data() {
    return {
      AutonnStatus
    };
  },

  computed: {
    displayData() {
      const buff = this.data;
      if (buff.project?.target?.target_image) {
        buff.project.target.target_image = "";
      }
      if (buff.project?.current_log) {
        buff.project.current_log = "";
      }

      return buff;
    }
  }
};
</script>
<style scoped>
.hyper {
  max-width: 370px;
}
@media screen and (min-width: 2150px) {
  .hyper {
    max-width: none;
  }
}
</style>
