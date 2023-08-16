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
    <div class="d-flex justify-center align-center pa-3">
      <v-img :src="item?.image" width="60" max-width="60" contain></v-img>
    </div>
    <div class="ml-3 d-flex flex-column justify-center" style="min-width: 115px">
      <div class="d-flex align-center" style="gap: 10px">
        <p style="color: #000000ff; letter-spacing: 1px; font-size: 14px" class="pa-0 ma-0">{{ item?.name }}</p>
      </div>
      <div class="d-flex justify-center mt-3 flex-column">
        <small style="color: #aaa; font-size: 10px; min-width: 70px; line-height: 1">
          {{ item?.create_date?.slice(0, 10) }}
        </small>
        <small style="color: #aaa; font-size: 10px; min-width: 110px; line-height: 1">
          created by
          <span style="color: #4a80ff">{{ item?.create_user }}</span>
        </small>
      </div>
    </div>
    <v-divider vertical class="ml-1 mr-3" />
    <v-hover v-slot="{ hover }">
      <div style="width: 100%; height: 100%">
        <v-slide-y-transition leave-absolute hide-on-leave>
          <div
            class="d-flex align-center"
            style="gap: 15px; height: 100%; transition-duration: 0.5s !important"
            v-if="!isMoreInfo ? true : !hover"
          >
            <div class="d-flex flex-column text-center" style="width: 70px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Info</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.info }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 70px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Memory</small>
              <p class="ma-0" style="font-size: 13px">{{ Number(item?.memory) }} MB</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 55px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">OS</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.os }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 40px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">CPU</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.cpu }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 70px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Accelerator</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.acc }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 55px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Engine</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.engine }}</p>
            </div>
          </div>
        </v-slide-y-transition>
        <v-slide-y-reverse-transition leave-absolute hide-on-leave>
          <!--  v-if="index % 2 === 0 ? false : hover" -->
          <div
            class="d-flex align-center"
            style="height: 100%; transition-duration: 0.5s !important; flex-flow: wrap"
            v-if="!isMoreInfo ? false : hover"
          >
            <div v-if="item?.host_ip" class="d-flex flex-column text-center mr-3" style="width: 125px">
              <small class="mb-0" style="letter-spacing: 1px; color: #aaa; font-size: 10px">IP Address</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.host_ip }}</p>
            </div>

            <div v-if="item?.host_port" class="d-flex flex-column text-center mr-3" style="width: 50px">
              <small class="mb-0" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Port</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.host_port }}</p>
            </div>

            <div v-if="item?.host_service_port" class="d-flex flex-column text-center mr-3" style="width: 80px">
              <small class="mb-0" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Service Port</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.host_service_port }}</p>
            </div>

            <div
              v-if="item?.info === 'k8s' && item?.nfs_ip"
              class="d-flex flex-column text-center mr-3"
              style="max-width: 120px"
            >
              <small class="mb-0" style="letter-spacing: 1px; color: #aaa; font-size: 10px">NFS IP</small>
              <p class="ma-0" style="font-size: 13px">{{ item?.nfs_ip }}</p>
            </div>

            <div v-if="item?.info === 'k8s' && item?.nfs_path" class="d-flex flex-column text-center mr-3">
              <small class="mb-0" style="letter-spacing: 1px; color: #aaa; font-size: 10px">NFS Path</small>
              <p class="ma-0" style="font-size: 13px">
                {{ item?.nfs_path }}
              </p>
            </div>
          </div>
        </v-slide-y-reverse-transition>
      </div>
    </v-hover>
  </div>
</template>
<script>
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
    return { prettyBytes };
  },

  computed: {
    isMoreInfo() {
      console.log(
        this.isValid(this.item?.host_ip) ||
          this.isValid(this.item?.host_port) ||
          this.isValid(this.item?.host_service_port) ||
          this.isValid(this.item?.nfs_ip) ||
          this.isValid(this.item?.nfs_path)
      );
      return (
        this.isValid(this.item?.host_ip) ||
        this.isValid(this.item?.host_port) ||
        this.isValid(this.item?.host_service_port) ||
        this.isValid(this.item?.nfs_ip) ||
        this.isValid(this.item?.nfs_path)
      );
    }
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

    isValid(data) {
      if (data !== "" && data !== undefined && data !== null) {
        return true;
      } else {
        return false;
      }
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
