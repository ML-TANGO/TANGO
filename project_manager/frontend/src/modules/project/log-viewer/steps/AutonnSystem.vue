<template>
  <StepContainer primaryColor="#0f8e9b" :isCompleted="isCompleted">
    <template #step-icon> <v-icon color="white">mdi-desktop-classic</v-icon> </template>
    <template #step-title> SYSTEM </template>
    <template #items>
      <StepItem titleColor="#71b5c5" contentColor="#aee2ea" v-for="(item, index) in displayData" :key="index">
        <template #title>
          <div>{{ item?.["devices"] || "" }}</div>
        </template>
        <template #content>
          <div>
            {{ item?.["gpu_model"] || "" }} <br />
            {{ displayMemory(item?.["memory"]) }}
          </div>
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
    displayData() {
      try {
        const gpus = JSON.parse(this.data?.["gpus"]);
        return gpus;
      } catch (err) {
        return [];
      }
    }
  },

  methods: {
    displayMemory(memory) {
      if (!memory) return "";

      if (memory.endsWith(")")) memory = memory.slice(0, memory.length - 1);
      return `${memory}G`;
    }
  }
};
</script>
<style></style>
