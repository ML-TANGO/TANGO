<template>
  <v-card style="height: 100%; overflow: none" class="px-10">
    <iframe :src="wedaDataSet" title="내용" width="100%" height="100%"></iframe>
  </v-card>
</template>
<script>
export default {
  data() {
    return {};
  },

  computed: {
    wedaDataSet() {
      if (process.env.NODE_ENV === "production") {
        const host = window.location.hostname;
        return `http://${host}:8086`;
      } else {
        return `${process.env.VUE_APP_ROOT_HOST}:8086`;
      }
    }
  }
};
</script>
<style lang="scss"></style>
