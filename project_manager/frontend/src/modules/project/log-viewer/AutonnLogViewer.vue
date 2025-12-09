<template>
  <div class="px-1">
    <v-tabs v-model="tab" color="#eee">
      <v-tabs-slider color="transparent"></v-tabs-slider>
      <v-tab :key="'train'" :style="{ backgroundColor: getColor(0) }" class="mr-0 tab-border"> Train </v-tab>
      <v-tab :key="'model'" :style="{ backgroundColor: getColor(1) }" class="ml-1 mr-0 tab-border"> Model </v-tab>
      <v-tab :key="'log'" :style="{ backgroundColor: getColor(2) }" class="ml-1 tab-border"> Log </v-tab>
    </v-tabs>

    <v-card class="pa-1 mt-2" color="#ddd">
      <v-tabs-items v-model="tab">
        <v-tab-item :key="'train'">
          <LogWithChart :data="data" />
        </v-tab-item>
        <v-tab-item :key="'model'">
          <ModelViewer />
        </v-tab-item>
        <v-tab-item :key="'log'">
          <LogWithText />
        </v-tab-item>
      </v-tabs-items>
    </v-card>
  </div>
</template>
<script>
import LogWithChart from "./LogWithChart.vue";
import LogWithText from "./LogWithText.vue";
import ModelViewer from "@/modules/project/model-viewer/ModelViewer.vue";
import { AutonnStatus } from "@/shared/enums";
export default {
  components: { LogWithChart, LogWithText, ModelViewer },

  props: {
    data: {
      default: null
    }
  },

  data() {
    return {
      tab: null,
      AutonnStatus,
      autoLogShown: false
    };
  },

  computed: {
    getColor() {
      return currTab => {
        if (this.tab === currTab) return "#4a80ff";
        else return "#eee";
      };
    }
  },

  watch: {
    // 학습 종료(train_end) 시 한 번만 Log 탭으로 전환(이후는 추론용 모델로 변환하는 과정)
    "data.progress": {
      handler(val) {
        const numericVal = Number(val);
        if (!Number.isNaN(numericVal) && !this.autoLogShown && numericVal >= AutonnStatus.TRAIN_END) {
          this.tab = 2;
          this.autoLogShown = true;
        }
      },
      immediate: true
    }
  }
};
</script>
<style lang="css" scoped>
.tab-border {
  border-radius: 4px;
}
</style>
