<template>
  <div>
    <div class="d-flex align-center">
      <div
        :style="{ backgroundColor: primaryColor }"
        style="border-radius: 8px; gap: 4px; color: white"
        class="d-flex align-center jutify-start px-4 py-2"
      >
        <slot name="step-icon"></slot>
        <slot name="step-title"></slot>
      </div>
      <slot name="step-description"></slot>
      <div class="ml-3">
        <v-icon v-if="isCompleted" size="38" :color="primaryColor">mdi-check-circle-outline</v-icon>
      </div>
    </div>

    <div class="d-flex justify-center align-center py-2" style="gap: 8px">
      <slot name="divider">
        <div style="height: 20px"></div>
      </slot>
    </div>

    <div class="d-flex flex-wrap" style="gap: 8px">
      <slot name="items"></slot>
    </div>
  </div>
</template>
<script>
export default {
  props: {
    primaryColor: {
      default: "#ffffff"
    },

    isCompleted: {
      default: false
    }
  }
};
</script>
<style></style>
