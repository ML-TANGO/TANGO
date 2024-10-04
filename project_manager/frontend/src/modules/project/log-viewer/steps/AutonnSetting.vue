<template>
  <StepContainer primaryColor="#6a913f" :isCompleted="isCompleted">
    <template #step-icon> <v-icon color="white">mdi-cog-outline</v-icon> </template>
    <template #step-title> SETTINGS </template>
    <template #divider>
      <div style="height: 2px; width: 100%; background-color: #6a913f"></div>
      <div style="color: #6a913f">HYPERPARAMETERS</div>
      <div style="height: 2px; width: 100%; background-color: #6a913f"></div>
    </template>
    <template #items>
      <StepItem titleColor="#81ac6f" contentColor="#abc89f" v-for="(item, index) in dispalyItems" :key="index">
        <template #title>
          <div>{{ item.title }}</div>
        </template>
        <template #content>
          <div>{{ item.content }}</div>
        </template>
      </StepItem>
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

    isCompleted: {
      default: false
    }
  },

  computed: {
    dispalyItems() {
      return [
        { title: "Weight Decay", content: this.fixedOrData(this.data?.["weight_decay"] || "") },
        { title: "Learning Rate", content: this.fixedOrData(this.data?.["lr0"] || "") },
        { title: "Momentum", content: this.fixedOrData(this.data?.["momentum"] || "") },
        { title: "Warmup Epochs", content: this.fixedOrData(this.data?.["warmup_epochs"] || "") },
        { title: "Warmup Blas LR", content: this.fixedOrData(this.data?.["warmup_bias_lr"] || "") },
        { title: "Warmup Momentum", content: this.fixedOrData(this.data?.["warmup_momentum"] || "") }
      ];
    }
  },

  methods: {
    fixedOrData(data, fixed = 5) {
      if (!isNaN(data)) {
        if (Number(data) === 0) return 0;
        return parseFloat(Number(data).toFixed(fixed).toLocaleString("ko-KR", { maximumFractionDigits: fixed }));
      } else {
        return data;
      }
    }
  }
};
</script>
<style></style>
