<template>
  <div style="width: 100%; height: 100%" ref="container">
    <v-stage :config="{ width: stageWidth, height: stageHeight }">
      <v-layer ref="layer" :config="{ offset: { x: 15, y: 0 } }">
        <v-group
          ref="bms"
          :config="{ x: stageWidth / 5, y: 95, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'bms')"
          @mouseover="onMouseover($event, 'bms')"
          @mouseout="onMouseout($event, 'bms')"
        >
          <v-line :config="createArrow(setStatus('bms', running))"></v-line>
          <v-text :config="{ x: 93, y: 32, text: 'BMS', ...configText }" />
        </v-group>
        <v-group
          ref="autoNN"
          :config="{ x: (stageWidth / 5) * 2, y: 95, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'yoloe')"
          @mouseover="onMouseover($event, 'autoNN')"
          @mouseout="onMouseout($event, 'autoNN')"
        >
          <v-line :config="createArrow(setStatus('autoNN', running))"></v-line>
          <v-text :config="{ x: 80, y: 32, text: 'Auto NN', ...configText }" />
        </v-group>
        <v-group
          ref="codeGen"
          :config="{ x: (stageWidth / 5) * 3, y: 95, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'codeGen')"
          @mouseover="onMouseover($event, 'codeGen')"
          @mouseout="onMouseout($event, 'codeGen')"
        >
          <v-line :config="createArrow(setStatus('codeGen', running))"></v-line>
          <v-text :config="{ x: 70, y: 32, text: 'Code Gen', ...configText }" />
        </v-group>
        <v-group
          ref="imageDepoly"
          :config="{ x: (stageWidth / 5) * 4, y: 95, offset: { x: 85, y: 40 } }"
          @mousedown="onMousedown($event, 'imageDepoly')"
          @mouseover="onMouseover($event, 'imageDepoly')"
          @mouseout="onMouseout($event, 'imageDepoly')"
        >
          <v-line :config="createArrow(setStatus('imageDepoly', running))"></v-line>
          <v-text :config="{ x: 65, y: 32, text: 'Image Depoly', ...configText }" />
        </v-group>
      </v-layer>
    </v-stage>
  </div>
</template>
<script>
import Konva from "konva";

export default {
  props: {
    running: {
      default: null
    }
  },

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
      dir: 1,

      orgStroke: "",

      runningOrder: ["bms", ["autoNN", "yoloe", "auto_nk"], "codeGen", "imageDepoly"]
      // currentIndex: -1
    };
  },

  watch: {
    runContainer(val) {
      console.log("val run container -- watcher= > ", val);
      this.period = 1;
      this.dir = 1;
      if (val === null) {
        this.animation.stop();
      } else {
        this.animation.start();
      }
    },

    running(val) {
      console.log("running watcher => ", val);
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
      if (this.runContainer) {
        this.runContainer.scale({ x: scale });
      }
    }, layer);

    this.animation.start();

    this.runContainer = null;
    //#endregion
  },

  methods: {
    onMousedown(event, container) {
      this.$emit("start", container);
    },

    onMouseover(event, container) {
      const node = this.$refs[container].getNode();
      this.orgStroke = node.children[0].getAttrs().stroke;

      console.log("this.orgStroke", this.orgStroke);

      node.children[0].setAttrs({
        stroke: "#000",
        strokeWidth: 2,
        strokeEnabled: true
      });
      node.getLayer().draw();

      document.body.style.cursor = "pointer";
    },

    onMouseout(event, container) {
      const node = this.$refs[container].getNode();
      node.children[0].setAttrs({
        stroke: this.orgStroke,
        strokeWidth: 2
      });
      node.getLayer().draw();
      document.body.style.cursor = "default";
      this.orgStroke = "";
    },

    setStatus(posotion, runningContainer) {
      const number = this.runningOrder.findIndex(q => q.includes(posotion));

      const currentIndex = this.runningOrder.findIndex(q => q.includes(runningContainer));

      console.log("currentIndex", currentIndex, "runningContainer", runningContainer);

      return {
        status: number < currentIndex ? "end" : number === currentIndex ? "running" : "preparing",
        posotion: posotion
      };
    },

    createArrow({ status, posotion }) {
      console.log("posotion", posotion);
      const x = 0;
      const y = 0;
      const fx = 196;
      const sx = 236;
      const tx = 40;

      const fy = 39;
      const sy = 76;

      const stroke = status === "end" ? "#cfd2cf" : status === "running" ? "#ff3d54" : "#4a80ff";
      const fill = status === "end" ? "#cfd2cf" : status === "running" ? "rgb(229, 190, 195)" : "#bfddff";

      if (status === "running") {
        this.$nextTick(() => {
          const ref = this.$refs[posotion];
          this.runContainer = ref.getNode();
        });
      }
      return {
        points: [x, y, x + fx, y, x + sx, y + fy, x + fx, y + sy, x, y + sy, x + tx, y + fy],
        fill: fill,
        stroke: stroke,
        strokeWidth: 2,
        closed: true,
        shadowColor: "black",
        shadowBlur: 5,
        shadowOffset: { x: 5, y: 5 },
        shadowOpacity: 0.5
      };
    }
  }
};
</script>
<style lang="scss" scoped></style>
