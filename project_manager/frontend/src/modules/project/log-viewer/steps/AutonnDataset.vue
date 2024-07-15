<template>
  <StepContainer primaryColor="#c00000" :isCompleted="isCompleted">
    <template #step-icon> <v-icon color="white">mdi-newspaper-variant-multiple</v-icon> </template>
    <template #step-title> Dataset </template>
    <template #items>
      <div class="text-center">
        <v-progress-circular
          v-if="trainValue < 100"
          :rotate="-90"
          :size="130"
          :width="10"
          :value="trainValue"
          color="#ef3e2b"
        >
          {{ trainValue.toFixed(2) }}%
        </v-progress-circular>
        <div v-else style="width: 130px; height: 130px; position: relative">
          <div style="width: 130px; height: 130px; position: absolute">
            <PieChart :chartData="trainChartDataset"> </PieChart>
          </div>
          <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%)">
            {{ trainValue.toFixed(2) }}%
          </div>
        </div>

        <div
          class="px-3 py-1 mt-4"
          style="border-radius: 100px; background-color: #ef3e2b; color: white; font-size: 14px"
        >
          CHECKING<br />
          TRAIN
        </div>
      </div>
      <div class="text-center">
        <v-progress-circular
          v-if="valValue < 100"
          :rotate="-90"
          :size="130"
          :width="10"
          :value="valValue"
          color="#ef3e2b"
        >
          {{ valValue.toFixed(2) }}%
        </v-progress-circular>
        <div v-else style="width: 130px; height: 130px; position: relative">
          <div style="width: 130px; height: 130px; position: absolute">
            <PieChart :chartData="valChartDataset"> </PieChart>
          </div>
          <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%)">
            {{ valValue.toFixed(2) }}%
          </div>
        </div>
        <div
          class="px-3 py-1 mt-4"
          style="border-radius: 100px; background-color: #ef3e2b; color: white; font-size: 14px"
        >
          CHECKING<br />
          VAL
        </div>
      </div>
    </template>
  </StepContainer>
</template>
<script>
import StepContainer from "../components/StepContainer.vue";
import PieChart from "@/modules/project/components/chart/PieChart.vue";

export default {
  components: { StepContainer, PieChart },

  props: {
    valDataset: {
      default: null
    },

    trainDataset: {
      default: null
    },

    isCompleted: {
      default: false
    }
  },

  data() {
    return {
      trainValue: 0,
      valValue: 0
    };
  },

  computed: {
    trainChartDataset() {
      const labels = ["found", "missing", "empty", "corrupted"];

      return {
        labels: labels,
        datasets: [
          {
            backgroundColor: ["#ef3e2b", "#9c9c9c", "#93bcd9", "#cc26ff"],
            data: labels.map(q => this.getDataOrZero(this.trainDataset, q))
          }
        ]
      };
    },
    valChartDataset() {
      const labels = ["found", "missing", "empty", "corrupted"];

      return {
        labels: labels,
        datasets: [
          {
            backgroundColor: ["#ef3e2b", "#9c9c9c", "#93bcd9", "#cc26ff"],
            data: labels.map(q => this.getDataOrZero(this.valDataset, q))
          }
        ]
      };
    }
  },

  watch: {
    valDataset() {
      const total = this.valDataset?.["total"] || 0;
      const current = this.valDataset?.["current"] || 0;
      if (total <= 0) {
        this.valValue = 0;
        return;
      }
      this.valValue = (current / total) * 100;
    },

    trainDataset() {
      const total = this.trainDataset?.["total"] || 0;
      const current = this.trainDataset?.["current"] || 0;
      if (total <= 0) {
        this.trainValue = 0;
        return;
      }
      this.trainValue = (current / total) * 100;
    }
  },

  methods: {
    getDataOrZero(object, key) {
      return object?.[key] || 0;
    }
  }
};
</script>
<style></style>
