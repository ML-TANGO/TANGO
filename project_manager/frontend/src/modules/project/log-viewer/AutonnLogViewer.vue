<template>
  <div class="px-1">
    <v-tabs v-model="tab" color="#eee">
      <v-tabs-slider color="transparent"></v-tabs-slider>
      <v-tab :key="'train'" :style="{ backgroundColor: getColor(0) }" class="mr-0 tab-border"> Train </v-tab>
      <v-tab :key="'model'" :style="{ backgroundColor: getColor(1) }" class="ml-1 tab-border"> Model </v-tab>
    </v-tabs>

    <v-card class="pa-1 mt-2" color="#ddd">
      <v-tabs-items v-model="tab">
        <v-tab-item :key="'train'">
          <LogWithChart :data="data" />
        </v-tab-item>
        <v-tab-item :key="'model'">
          <ModelViewer />
        </v-tab-item>
      </v-tabs-items>
    </v-card>
  </div>
</template>
<script>
import LogWithChart from "./LogWithChart.vue";
import ModelViewer from "@/modules/project/model-viewer/ModelViewer.vue";
export default {
  components: { LogWithChart, ModelViewer },

  props: {
    data: {
      default: null
    }
  },

  data() {
    return {
      tab: null
    };
  },

  computed: {
    getColor() {
      return currTab => {
        if (this.tab === currTab) return "#4a80ff";
        else return "#eee";
      };
    }
  }
};
</script>
<style lang="css" scoped>
.tab-border {
  border-radius: 4px;
}
</style>
