<template>
  <div class="my-3">
    <div class="d-flex align-center justify-center">
      <v-btn @click="isBanner = !isBanner" icon style="margin-top: -3px">
        <v-icon v-if="!isBanner">mdi-chevron-down</v-icon>
        <v-icon v-else>mdi-chevron-up</v-icon>
      </v-btn>
      <div class="mr-3" style="font-size: 13px">
        <span style="color: #555">{{ title }}</span> <span style="color: #ccc" v-if="count >= 0">({{ count }})</span>
      </div>
      <v-divider></v-divider>
      <slot name="action"></slot>
    </div>
    <v-banner v-model="isBanner" single-line transition="slide-y-transition" class="pl-3">
      <div class="pa-3 pt-5">
        <slot name="content"> </slot>
      </div>
    </v-banner>
  </div>
</template>
<script>
export default {
  props: {
    title: {
      default: ""
    },
    count: {
      default: 0
    }
  },

  data() {
    return {
      isBanner: true
    };
  }
};
</script>
<style lang=""></style>
