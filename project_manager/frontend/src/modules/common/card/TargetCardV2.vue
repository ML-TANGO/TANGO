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
    <div class="ml-3 d-flex flex-column justify-center" style="min-width: 115px; max-width: 115px">
      <div class="d-flex align-center" style="gap: 10px">
        <p style="color: #000000ff; letter-spacing: 1px; font-size: 14px" class="pa-0 ma-0">{{ item?.name }}</p>
      </div>
      <div class="d-flex justify-center mt-3 flex-column">
        <small style="color: #aaa; font-size: 10px; min-width: 70px; line-height: 1">
          {{ item?.create_date?.slice(0, 10) }}
        </small>
        <small style="color: #aaa; font-size: 10px; min-width: 110px; line-height: 1" v-if="item?.create_user !== ''">
          created by
          <span style="color: #4a80ff">{{ item?.create_user }}</span>
        </small>
      </div>
    </div>
    <v-divider vertical class="ml-1 mr-3" />
    <div style="width: 100%; height: 100%; position: relative; overflow: hidden">
      <div :ref="`${item?.info}_${item.id}`" style="position: absolute; height: 100%; white-space: nowrap">
        <span class="item-wrapper">
          <small class="item-title">Info</small>
          <p class="item-value">{{ item?.info }}</p>
        </span>

        <span class="item-wrapper">
          <small class="item-title">Memory</small>
          <p class="item-value">{{ Number(item?.memory) }} MB</p>
        </span>

        <span class="item-wrapper">
          <small class="item-title">OS</small>
          <p class="item-value">{{ item?.os }}</p>
        </span>

        <span class="item-wrapper">
          <small class="item-title">CPU</small>
          <p class="item-value">{{ item?.cpu }}</p>
        </span>

        <span class="item-wrapper">
          <small class="item-title">Accelerator</small>
          <p class="item-value">{{ item?.acc }}</p>
        </span>

        <span class="item-wrapper">
          <small class="item-title">Engine</small>
          <p class="item-value">{{ item?.engine }}</p>
        </span>

        <span v-if="item?.host_ip" class="item-wrapper">
          <small class="item-title">IP Address</small>
          <p class="item-value">{{ item?.host_ip }}</p>
        </span>

        <span v-if="item?.host_port" class="item-wrapper">
          <small class="item-title">Port</small>
          <p class="item-value">{{ item?.host_port }}</p>
        </span>

        <span v-if="item?.host_service_port" class="item-wrapper">
          <small class="item-title">Service Port</small>
          <p class="item-value">{{ item?.host_service_port }}</p>
        </span>

        <span v-if="isK8S(item?.info) && item?.nfs_ip" class="item-wrapper">
          <small class="item-title">NFS IP</small>
          <p class="item-value">{{ item?.nfs_ip }}</p>
        </span>

        <span v-if="isK8S(item?.info) && item?.nfs_path" class="item-wrapper">
          <small class="item-title">NFS Path</small>
          <p class="item-value">{{ item?.nfs_path }}</p>
        </span>
      </div>
    </div>
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
      return (
        this.isValid(this.item?.host_ip) ||
        this.isValid(this.item?.host_port) ||
        this.isValid(this.item?.host_service_port) ||
        this.isValid(this.item?.nfs_ip) ||
        this.isValid(this.item?.nfs_path)
      );
    }
  },

  watch: {
    item: {
      immediate: true,
      deep: true,
      handler() {
        this.setAnimation();
      }
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
    },

    isK8S(info) {
      if (!info) return false;
      return info.toLowerCase().includes("k8s");
    },

    setAnimation() {
      this.$nextTick(() => {
        const ref = this.$refs[`${this.item?.info}_${this.item.id}`];
        if (ref.parentNode.clientWidth < ref.clientWidth) ref.classList.add("animation");
        else {
          ref.classList.remove("animation");
        }
      });
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

<style lang="css" scoped>
.animation {
  /* 
        animation-name: left-right;
        animation-duration: 3s es;
        animation-iteration-count: infinite;
        animation-direction: alternate; */
  animation: left-right 8s infinite alternate linear;
}

@keyframes left-right {
  0%,
  30% {
    transform: translateX(0%);
    left: 0%;
  }

  50% {
    transform: translateX(-50%);
    left: 50%;
  }

  70%,
  100% {
    transform: translateX(-100%);
    left: 100%;
  }
}

.item-wrapper {
  height: 100%;

  display: inline-flex;
  flex-direction: column;
  margin: 0px 12px;

  text-align: center;
  justify-content: center;
}

.item-title {
  letter-spacing: 1px;
  color: #aaa;
  font-size: 10px;
  margin-bottom: 8px;
}

.item-value {
  font-size: 13px;
  margin: 0px;
}
</style>
