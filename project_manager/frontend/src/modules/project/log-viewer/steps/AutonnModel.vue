<template>
  <StepContainer primaryColor="#5d239d" :isCompleted="isCompleted">
    <template #step-icon> <v-icon color="white">mdi-calculator-variant-outline</v-icon> </template>
    <template #step-title> MODEL </template>
    <template #step-description> <div class="pl-3">YOLOV7 - TINY</div> </template>
    <template #items>
      <StepItem titleColor="#9363b7" contentColor="#b797cf" v-for="(item, index) in dispalyItems" :key="index">
        <template #title>
          <div>{{ item.title }}</div></template
        >
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
        { title: "Layers", content: this.data?.["layers"]?.toLocaleString() || "" },
        { title: "Params", content: this.data?.["parameters"]?.toLocaleString() || "" },
        { title: "Grads", content: this.data?.["gradients"]?.toLocaleString() || "" },
        { title: "Flops", content: this.data?.["flops"]?.toLocaleString() || "" }
      ];
    }
  }
};
</script>
<style></style>
