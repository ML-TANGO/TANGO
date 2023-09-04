<template>
  <!-- style="width: calc(80% / 4)" -->
  <div class="container" @mouseout="onMouseout" @mouseover="onMouseover" @click="onClick">
    <v-img
      :src="progressIcon"
      max-height="95"
      contain
      style="filter: drop-shadow(5px 5px 5px #666)"
      :class="status === 'running' ? 'run-container' : ''"
    ></v-img>
    <div style="height: 100%" class="d-flex justify-center align-center text">{{ text }}</div>
  </div>
</template>
<script>
import DefaultIcon from "@/assets/progress/default.png";
import DefaultHoverIcon from "@/assets/progress/default_hover.png";

import RunningIcon from "@/assets/progress/running.png";
import RunningHoverIcon from "@/assets/progress/running_hover.png";

import CompleteIcon from "@/assets/progress/complete.png";
import CompleteHoverIcon from "@/assets/progress/complete_hover.png";
export default {
  props: {
    text: {
      default: ""
    },

    status: {
      default: "preparing"
    }
  },

  data() {
    return {
      isHover: false
    };
  },

  computed: {
    progressIcon() {
      if (this.isHover) {
        if (this.status === "running") {
          return RunningHoverIcon;
        } else if (this.status === "completed") {
          return CompleteHoverIcon;
        } else {
          return DefaultHoverIcon;
        }
      } else {
        if (this.status === "running") {
          return RunningIcon;
        } else if (this.status === "completed") {
          return CompleteIcon;
        } else {
          return DefaultIcon;
        }
      }
    }
  },

  methods: {
    onClick(e) {
      this.$emit("click", e);
    },

    onMouseout() {
      this.isHover = false;
    },

    onMouseover() {
      this.isHover = true;
    }
  }
};
</script>
<style lang="scss" scoped>
.container {
  position: relative;
  .text {
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0px;
    left: 3px;
    font-size: 0.9vw;
  }

  &:hover {
    cursor: pointer !important;
  }
}

.run-container {
  /* animation-name: run;
  animation-duration: 3s;
  animation-iteration-count: infinite;
  animation-direction: reverse;
  animation-timing-function: linear;
  animation-fill-mode: none;
  animation-delay: 5s;
  animation-fill-mode: forwards; */

  animation: run 1.5s 1s infinite ease-in-out;
}

@keyframes run {
  0% {
    transform: scaleX(1);
  }
  50% {
    transform: scaleX(1.08);
  }
  100% {
    transform: scaleX(1);
  }
}
</style>
