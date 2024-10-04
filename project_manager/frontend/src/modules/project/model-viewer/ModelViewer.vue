<template>
  <v-card style="height: 920px; overflow: none" class="">
    <iframe :src="modelSrc" title="내용" width="100%" height="100%"></iframe>
  </v-card>
</template>
<script>
export default {
  computed: {
    modelSrc() {
      if (process.env.NODE_ENV === "production") {
        const host = window.location.hostname;
        return `http://${host}:8100`;
      } else {
        return `${process.env.VUE_APP_ROOT_HOST}:8100`;
      }
    }
  }
};
</script>
<style lang=""></style>
