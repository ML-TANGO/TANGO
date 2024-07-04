<template>
  <div style="padding: 8px">
    <div v-for="([key, value], index) in Object.entries(displayData)" :key="index" class="py-2">
      <div class="mb-1">update_id : {{ key }}</div>
      <div style="white-space: normal">{{ value }}</div>
      <v-divider></v-divider>
    </div>
  </div>
</template>
<script>
export default {
  props: {
    data: {
      default: null
    }
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
<style></style>
