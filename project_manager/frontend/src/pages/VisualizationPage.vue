<template>
  <v-card style="height: 100%; overflow: none" class="px-10">
    <iframe :src="HongIKVis2Code" title="내용" width="100%" height="100%"></iframe>
  </v-card>
</template>
<script>
export default {
  data() {
    return {};
  },

  computed: {
    HongIKVis2Code() {
      if (process.env.NODE_ENV === "production") {
        const host = window.location.hostname;
        return `http://${host}:8091`;
      } else {
        return `${process.env.VUE_APP_ROOT_HOST}:8091`;
      }
    }
  }
};
</script>
<style lang="scss"></style>
