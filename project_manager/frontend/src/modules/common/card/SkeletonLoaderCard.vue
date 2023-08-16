<template>
  <div
    id="card-list-item"
    class="ma-3 d-flex pa-3 align-center v-skeleton-loader__bone"
    style="height: 100px"
    :style="{
      background: '#EFEFEF'
    }"
  >
    <div class="d-flex justify-center align-center pa-3">
      <span style="width: 60px; height: 60px; background: #ccc; border-radius: 4px"></span>
    </div>
    <div class="ml-3 d-flex flex-column justify-center" style="width: 240px">
      <div class="d-flex align-center" style="gap: 10px">
        <p style="color: #000000ff; letter-spacing: 1px; font-size: 14px" class="pa-0 ma-0">
          <v-chip style="font-size: 8px; height: 20px; color: white; width: 150px; background: #aaa">
            <!-- {{ ObjectType[item?.OBJECT_TYPE] }} -->
          </v-chip>
        </p>
      </div>
      <div class="d-flex align-center mt-3">
        <small style="color: #aaa; font-size: 11px">{{ item?.UPT_DTM?.slice(0, 10) }}</small>
        <small style="color: #aaa; font-size: 11px">
          <v-chip style="font-size: 8px; height: 20px; color: white; width: 150px; background: #ccc">
            <!-- {{ ObjectType[item?.OBJECT_TYPE] }} -->
          </v-chip>
        </small>
      </div>
    </div>
    <div class="d-flex align-center" style="gap: 60px; height: 100%">
      <div class="d-flex flex-column text-center">
        <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 11px"></small>
        <p class="ma-0" style="font-size: 14px"></p>
      </div>
      <div class="d-flex flex-column text-center">
        <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 11px"></small>
        <p class="ma-0" style="font-size: 14px"></p>
      </div>
      <div class="d-flex flex-column text-center">
        <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 11px"></small>
        <p class="ma-0" style="font-size: 14px"></p>
      </div>
    </div>
  </div>
</template>
<script>
import { ObjectType, DataType } from "@/shared/enums";
export default {
  props: {
    isSelected: {
      default: false
    },

    isHover: {
      default: false
    },

    item: {
      default: () => ({ ObjectType, DataType })
    }
  },
  data() {
    return {};
  },

  methods: {
    onMouseover() {
      this.$emit("mouseover");
    },

    onMouseleave() {
      this.$emit("mouseleave");
    },

    onSelected() {
      this.$emit("click");
    }
  }
};
</script>
<style lang="scss" scoped>
.v-skeleton-loader__bone:after {
  -webkit-animation: loading 1.5s infinite;
  animation: loading 1.5s infinite;
  content: "";
  height: 100%;
  left: 0;
  position: absolute;
  right: 0;
  top: 0;
  transform: translateX(-100%);
  z-index: 1;
}
</style>
