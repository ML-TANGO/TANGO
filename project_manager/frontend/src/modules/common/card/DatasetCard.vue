<template>
  <div
    id="card-list-item"
    class="ma-3 d-flex pa-3 align-center"
    style="height: 100px"
    :class="isSelected ? 'select' : ''"
    :style="{
      background: isSelected ? 'rgba(255, 61, 84, 0.2)' : isHover ? '#EFEFEF' : '#F7F7F7',
      borderRadius: isSelected ? '0px' : '4px'
    }"
    @click="onSelected"
    @mouseover="onMouseover"
    @mouseleave="onMouseleave"
  >
    <!-- <div class="d-flex justify-center align-center pa-3">
      <v-img :src="getDatasetImage(item?.THUM_NAIL)" width="60" max-width="60" contain></v-img>
    </div> -->
    <div class="ml-3 d-flex flex-column justify-center" style="width: 380px">
      <div class="d-flex align-center" style="gap: 10px">
        <p style="color: #4a80ff; letter-spacing: 1px; font-size: 18px; font-weight: bold" class="pa-0 ma-0">
          {{ item?.name }}
        </p>
        <v-img v-if="item?.thumbnail" :src="item?.thumbnail" max-height="50" max-width="215" contain></v-img>

        <!-- <v-chip
          style="font-size: 8px; height: 20px; color: white"
          :style="{ backgroundColor: isSelected ? '#25c0dc' : '#25c0dc' }"
        >
          {{ DataType[item?.DATA_TYPE] }}
        </v-chip>
        <v-chip
          style="font-size: 8px; height: 20px; color: white"
          :style="{ backgroundColor: isSelected ? '#25c0dc' : '#25c0dc' }"
        >
          {{ ObjectType[item?.OBJECT_TYPE] }}
        </v-chip> -->
      </div>
      <div class="d-flex align-center mt-3" style="gap: 10px">
        <!-- <small style="color: #aaa; font-size: 11px">폴더 명 </small>
        <p style="color: #4a80ff; letter-spacing: 1px; font-size: 18px; font-weight: bold" class="pa-0 ma-0">
          {{ item?.name }}
        </p> -->
        <small style="color: #aaa; font-size: 11px">폴더 생성일</small>
        <small style="color: #000; font-size: 11px">{{ item?.creation_time }}</small>
        <!-- <small style="color: #aaa; font-size: 11px">마지막 수정 날짜</small>
        <small style="color: #000; font-size: 11px">{{ item?.last_modified_time?.slice(0, 10) }}</small> -->
        <!-- <small style="color: #aaa; font-size: 11px">
          created by
          <span style="color: #4a80ff">{{ item?.CRN_USR }}</span>
        </small> -->
      </div>
    </div>
    <v-divider vertical class="ml-11 mr-10" />
    <div class="d-flex align-center" style="gap: 60px; height: 100%">
      <div class="d-flex flex-column text-center">
        <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 11px">FILES</small>
        <p class="ma-0" style="font-size: 14px">{{ item?.file_count || "-" }}</p>
      </div>
      <div class="d-flex flex-column text-center">
        <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 11px">SIZE</small>
        <p class="ma-0" style="font-size: 14px">{{ item?.size ? prettyBytes(item?.size) : "-" }}</p>
      </div>
      <!-- <div class="d-flex flex-column text-center">
        <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 11px">CLASS</small>
        <p class="ma-0" style="font-size: 14px">{{ item?.CLASS_COUNT || "-" }}</p>
      </div> -->
    </div>
  </div>
</template>
<script>
import { ObjectType, DataType } from "@/shared/enums";

import prettyBytes from "pretty-bytes";
export default {
  props: {
    isSelected: {
      default: false
    },

    isHover: {
      default: false
    },

    item: {
      default: () => ({})
    }
  },
  data() {
    return { prettyBytes, ObjectType, DataType };
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
    },

    getDatasetImage(value) {
      const host = window.location.hostname;
      const imageAddress = "http://" + host + ":8095" + value;

      return imageAddress;
    }
  }
};
</script>
<style lang="scss" scoped>
@mixin magic-border($width, $color, $duration, $direction) {
  position: relative;
  &:before {
    content: "";
    position: absolute;
    width: calc(100% + #{$width * 2});
    height: calc(100% + #{$width * 2});
    top: calc(#{$width}/ -1);
    left: calc(#{$width}/ -1);
    background: linear-gradient(to right, $color 0%, $color 100%), linear-gradient(to top, $color 50%, transparent 50%),
      linear-gradient(to top, $color 50%, transparent 50%), linear-gradient(to right, $color 0%, $color 100%),
      linear-gradient(to left, $color 0%, $color 100%);
    background-size: 100% $width, $width 200%, $width 200%, 0% $width, 0% $width;
    background-position: 50% 100%, 0% 0%, 100% 0%, 100% 0%, 0% 0%;
    background-repeat: no-repeat, no-repeat;
    transition: transform $duration ease-in-out, background-position $duration ease-in-out,
      background-size $duration ease-in-out;
    transform: scaleX(0) rotate(180deg * $direction);
    transition-delay: $duration * 2, $duration, 0s;

    border-radius: 4px;
  }
  &.select {
    &:before {
      background-size: 200% $width, $width 400%, $width 400%, 55% $width, 55% $width;
      background-position: 50% 100%, 0% 100%, 100% 100%, 100% 0%, 0% 0%;
      transform: scaleX(1) rotate(180deg * $direction);
      transition-delay: 0s, $duration, $duration * 2;
    }
    transition: 0.4s;
    transition-delay: 0.6s;
  }
}

#card-list-item {
  @include magic-border(2px, rgb(255, 61, 84), 0.2s, 1);
}
</style>
