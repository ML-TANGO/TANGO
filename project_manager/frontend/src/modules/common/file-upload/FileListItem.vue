<template>
  <div
    style="width: 100%; height: 100%"
    class="d-flex"
    :class="isImage ? 'align-center justify-center' : 'mt-3 align-start justify-space-between pl-6'"
  >
    <slot></slot>
    <v-tooltip bottom>
      <template v-slot:activator="{ on, attrs }">
        <v-btn :class="isImage ? 'delete-btn' : ''" icon v-bind="attrs" v-on="on" @click="removeFile">
          <v-icon>mdi-file-document-remove</v-icon>
        </v-btn>
      </template>
      <span style="font-size: 10px">삭제</span>
    </v-tooltip>
  </div>
</template>

<script>
export default {
  props: {
    isImage: {
      default: true
    }
  },
  mounted() {
    this.$emit("mounted");
  },

  methods: {
    removeFile() {
      this.$emit("remove");
    }
  }
};
</script>

<style lang="scss" scoped>
.delete-btn {
  position: absolute;
  top: 5px;
  right: 5px;
}
</style>
