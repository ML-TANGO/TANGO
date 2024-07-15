<template>
  <div class="d-flex flex-column">
    <div
      class="px-4 round-rect d-flex align-center justify-center"
      :style="{ backgroundColor: titleColor }"
      style="border: 3px solid #ffffff; margin-bottom: -25px; z-index: 3"
    >
      <div
        style="height: 40px; width: 80px; color: white; font-weight: bold"
        class="default-item d-flex align-center justify-center"
      >
        <slot name="title"></slot>
      </div>
    </div>
    <div
      class="default-item d-flex align-center justify-center"
      style="height: 70px; padding-top: 25px; border-radius: 0px 0px 16px 16px"
      :style="{ backgroundColor: contentColor }"
    >
      <slot name="content"></slot>
    </div>
  </div>
</template>
<script>
export default {
  props: {
    titleColor: {
      default: "#ffffff"
    },

    contentColor: {
      default: "#ffffff"
    }
  }
};
</script>
<style lang="css">
.round-rect {
  border-radius: 9999px;
}

.default-item {
  white-space: break-spaces;
  text-align: center;

  font-size: 12px;
}
</style>
