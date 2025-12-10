<template>
  <StepContainer class="model-typography" primaryColor="#5d239d" :isCompleted="isCompleted">
    <template #step-icon> <v-icon color="white">mdi-calculator-variant-outline</v-icon> </template>
    <template #step-title> MODEL </template>
    <template #step-description>
      <div class="pl-3 model-desc">{{ baseModel?.["model_name"] || "" }} {{ baseModel?.["model_size"] || "" }}</div>
    </template>
    <template #items>
      <div class="item-grid">
        <div class="item-row">
          <StepItem
            titleColor="#9363b7"
            contentColor="#b797cf"
            v-for="(item, index) in firstRowItems"
            :key="`row1-${index}`"
          >
            <template #title>
              <div>{{ item.title }}</div>
            </template>
            <template #content>
              <div>{{ item.content }}</div>
            </template>
          </StepItem>
        </div>
        <div class="item-row">
          <StepItem
            titleColor="#9363b7"
            contentColor="#b797cf"
            v-for="(item, index) in secondRowItems"
            :key="`row2-${index}`"
          >
            <template #title>
              <div>{{ item.title }}</div>
            </template>
            <template #content>
              <div>{{ item.content }}</div>
            </template>
          </StepItem>
        </div>
      </div>
    </template>
  </StepContainer>
</template>
<script>
import StepContainer from "../components/StepContainer.vue";
import StepItem from "../components/StepItem.vue";
export default {
  components: { StepContainer, StepItem },

  props: {
    data: {
      default: null
    },

    batchSize: {
      default: null
    },

    arguments: {
      default: null
    },

    baseModel: {
      default: null
    },

    isCompleted: {
      default: false
    }
  },

  computed: {
    dispalyItems() {
      return [
        { title: "Layers", content: this.data?.["layers"]?.toLocaleString() || "" },
        { title: "Params", content: this.data?.["parameters"]?.toLocaleString() || "" },
        { title: "Grads", content: this.data?.["gradients"]?.toLocaleString() || "" },
        { title: "GFLOPS", content: this.data?.["flops"]?.toLocaleString() || "" },
        { title: "Batch Size", content: this.batchSizeText },
        { title: "Image Size", content: this.imageSizeText }
      ];
    },

    firstRowItems() {
      return this.dispalyItems.slice(0, 4);
    },

    secondRowItems() {
      return this.dispalyItems.slice(4);
    },

    batchSizeText() {
      const source = this.batchSize || this.data?.["batch_size"] || {};
      const low = this.asNumber(source?.low);
      const high = this.asNumber(source?.high);
      const finalPerDevice = this.computePerDeviceFinal();
      const rawMax = high;

      // Final state: show only concise per-GPU with margin
      if (finalPerDevice !== null) {
        const marginPct = rawMax
          ? Math.round((1 - finalPerDevice / rawMax) * 100)
          : null;
        const marginText =
          marginPct !== null && isFinite(marginPct) ? `\n(${marginPct}% margin)` : "";
        return `${this.formatNumber(finalPerDevice)}/GPU${marginText}`;
      }

      // Searching state: follow original low/high messaging
      if (low === null && high === null) return "Auto search pending";
      if (low !== null && high === null) return `Testing up to ${this.formatNumber(low)}`;
      if (low !== null && high !== null) {
        const label = high - low <= 1 ? "Locked" : "Tuning";
        let value = `${this.formatNumber(low)} ~ ${this.formatNumber(high)}`;
        if (label === "Locked") value = `${this.formatNumber(low)}`;
        return `${label}: ${value}`;
      }
      if (low === null && high !== null) return `Fallback from ${this.formatNumber(high)}`;
      return "";
    },

    imageSizeText() {
      const imgs = this.arguments?.["img_size"] || this.data?.["img_size"];
      if (!imgs || !Array.isArray(imgs)) return "Waiting...";

      const values = imgs.map(v => this.asNumber(v)).filter(v => v !== null);
      if (values.length === 0) return "Waiting...";
      if (values.length === 1) return `${values[0]} x ${values[0]}`;
      const [train, test] = values;
      if (train && test) {
        return `${train} x ${test}`;
      }
      if (train) return `${train} x ${train}`;
      if (test) return `${test} x ${test}`;
      return "Waiting...";
    }
  },

  methods: {
    asNumber(value) {
      const num = Number(value);
      return Number.isFinite(num) ? num : null;
    },

    formatNumber(value) {
      const num = this.asNumber(value);
      return num === null ? "" : num.toLocaleString();
    },

    computePerDeviceFinal() {
      const args = this.arguments || {};
      const total = this.asNumber(args["total_batch_size"] || args["batch_size"]);
      const world = this.asNumber(args["world_size"]);

      if (total !== null && total > 0 && world !== null && world > 0) {
        return total / world;
      }
      return null;
    },

    computeGlobalFinal(perDevice) {
      const args = this.arguments || {};
      const total = this.asNumber(args["total_batch_size"] || args["batch_size"]);
      if (total !== null && total > 0) return total;
      if (perDevice !== null) {
        const world = this.asNumber(args["world_size"]);
        if (world !== null) return perDevice * world;
      }
      return null;
    }
  }
};
</script>
<style scoped>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@600;700&display=swap");

.model-typography :deep(.default-item),
.model-desc {
  font-family: "IBM Plex Sans", "Noto Sans KR", "Helvetica Neue", Arial, sans-serif;
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum";
  font-weight: 700;
  font-size: 14px;
  letter-spacing: 0.2px;
}

.item-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  width: 100%;
}

.item-grid {
  display: grid;
  gap: 8px;
  width: 100%;
}
</style>
