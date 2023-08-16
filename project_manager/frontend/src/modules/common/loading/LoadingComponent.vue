<template lang="">
  <v-overlay :value="loading" style="z-index: 999">
    <v-progress-circular :indeterminate="loading" color="primary"></v-progress-circular>
  </v-overlay>
</template>
<script>
export default {
  props: {
    loading: {
      default: false
    }
  }
};
</script>
<style lang=""></style>
