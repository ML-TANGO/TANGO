<template>
  <div style="width: 100%; height: 100%" ref="container">
    <v-stage :config="{ width: stageWidth, height: stageHeight }">
      <v-layer ref="layer" :config="{ offset: { x: 15, y: 0 } }">
        <v-group
          ref="bms"
          :config="{ x: 450, y: 75, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'bms')"
        >
          <v-line :config="createArrow('end')"></v-line>
          <v-text :config="{ x: 75, y: 32, text: 'BMS', ...configText }" />
        </v-group>
        <v-group
          ref="autoNN"
          :config="{ x: 450 + 190, y: 75, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'autoNN')"
        >
          <v-line :config="createArrow('running')"></v-line>
          <v-text :config="{ x: 60, y: 32, text: 'Auto NN', ...configText }" />
        </v-group>
        <v-group
          ref="imageGen"
          :config="{ x: 450 + 190 + 190, y: 75, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'imageGen')"
        >
          <v-line :config="createArrow('preparing')"></v-line>
          <v-text :config="{ x: 50, y: 32, text: 'Image Gen', ...configText }" />
        </v-group>
        <v-group
          ref="imageDepoly"
          :config="{ x: 450 + 190 + 190 + 190, y: 75, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'imageDepoly')"
        >
          <v-line :config="createArrow('preparing')"></v-line>
          <v-text :config="{ x: 45, y: 32, text: 'Image Depoly', ...configText }" />
        </v-group>
        <v-group
          ref="runImage"
          :config="{ x: 450 + 190 + 190 + 190 + 190, y: 75, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'runImage')"
        >
          <v-line :config="createArrow('preparing')"></v-line>
          <v-text :config="{ x: 55, y: 32, text: 'Run Image', ...configText }" />
        </v-group>
      </v-layer>
    </v-stage>
  </div>
</template>
<script>
import Konva from "konva";

export default {
  data() {
    return {
      //   configStage: { width: 200, height: 180 },
      configText: {
        fontSize: 18,
        fontFamily: "NEXON Lv1 Gothic Low OTF",
        fill: "#363434"
      },
      stageWidth: 180,
      stageHeight: 180,

      // animation property
      runContainer: null,
      animation: null,
      period: 1,
      dir: 1
    };
  },

  watch: {
    runContainer(val) {
      this.period = 1;
      this.dir = 1;
      if (val === null) {
        this.animation.stop();
      } else {
        this.animation.start();
      }
    }
  },

  mounted() {
    const container = this.$refs.container;
    this.stageWidth = container.offsetWidth;
    this.stageHeight = container.offsetHeight;

    //#region animation load............
    const layer = this.$refs.layer.getNode();
    this.animation = new Konva.Animation(() => {
      const scale = this.period + 0.003 * this.dir;
      if (scale > 1.08) this.dir = -1;
      else if (scale < 1) this.dir = 1;
      this.period = scale;
      if (this.runContainer) this.runContainer.scale({ x: scale, y: scale });
    }, layer);

    this.animation.start();
    //#endregion
  },

  methods: {
    onMousedown(event, container) {
      const ref = this.$refs[container];
      if (this.runContainer) {
        this.runContainer.scale({ x: 1, y: 1 });
      }
      this.runContainer = ref.getNode();
    },

    createArrow(status) {
      const x = 0;
      const y = 0;
      const fx = 176;
      const sx = 216;
      const tx = 40;

      const fy = 39;
      const sy = 76;

      const stroke = status === "end" ? "#cfd2cf" : status === "running" ? "#ff3d54" : "#4a80ff";
      const fill = status === "end" ? "#cfd2cf" : status === "running" ? "rgba(255, 61, 84, 0.2)" : "#4a80ff";
      return {
        points: [x, y, x + fx, y, x + sx, y + fy, x + fx, y + sy, x, y + sy, x + tx, y + fy],
        fill: fill,
        stroke: stroke,
        strokeWidth: 2,
        closed: true
      };
    }
  }
};
</script>
<style lang="scss" scoped></style>
