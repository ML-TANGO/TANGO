<template>
  <div style="width: 100%; height: 100%">
    <div
      class="pa-2 d-flex justify-center"
      style="
        margin: 3px;
        outline: 3px solid white;
        border-radius: 100px;

        color: white;
        margin-bottom: -25px;
        font-size: 14px;
      "
      :style="{ backgroundColor: primaryColor }"
    >
      {{ title }}
    </div>
    <!-- :style="{ border: `1px solid ${primaryColor}` }" border-radius: 8px; padding-top: 25px -->
    <slot name="chart-area"></slot>
  </div>
</template>
<script>
export default {
  components: {},
  props: {
    primaryColor: {
      default: "#ffffff"
    },

    title: {
      default: ""
    }
  }
};
</script>
<style></style>
