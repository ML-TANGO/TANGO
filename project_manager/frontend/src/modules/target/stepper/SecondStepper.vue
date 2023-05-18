<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <p class="text-h5 mb-4" style="color: #4a80ff">Target Specification</p>

      <!-- Target Info -->
      <div class="d-flex align-center mt-11" style="gap: 25px">
        <div style="width: 150px">Target Info</div>
        <v-radio-group :value="targetInfo" row hide-details="" class="ma-0" @change="infoChange">
          <v-radio label="PC" value="pc"></v-radio>
          <v-radio label="OnDevice" value="ondevice"></v-radio>
          <v-radio label="Cloud" value="cloud"></v-radio>
        </v-radio-group>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <!-- Engine -->
      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">Engine</div>
        <v-radio-group :value="engine" row hide-details="" class="ma-0" @change="engineChange">
          <v-radio label="ACL" value="acl"></v-radio>
          <v-radio label="RKNN" value="rknn"></v-radio>
          <v-radio label="Pytorch" value="pytorch"></v-radio>
        </v-radio-group>
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <!-- info -->
      <div class="d-flex" style="gap: 25px" v-if="true">
        <div style="min-width: 150px">Deploy Config</div>
        <div class="d-flex flex-column" style="gap: 25px; width: 100%">
          <div class="d-flex" style="gap: 10px">
            <v-autocomplete
              :value="os"
              :items="osItem"
              label="OS"
              outlined
              dense
              hide-details
              item-text="label"
              @change="osChange"
            />
            <v-autocomplete
              :value="cpu"
              :items="cpuItem"
              label="CPU"
              outlined
              dense
              hide-details
              item-text="label"
              @change="cpuChange"
            />
          </div>

          <div class="d-flex" style="gap: 10px">
            <v-autocomplete
              :value="acc"
              :items="accItem"
              label="Accelerator"
              outlined
              dense
              hide-details
              style="width: 50%"
              item-text="label"
              @change="accChange"
            />
            <v-text-field
              :value="memory"
              type="number"
              outlined
              dense
              label="Memory"
              hide-details
              style="width: 50%"
              @change="memoryChange"
            />
          </div>
        </div>
      </div>
    </div>
    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="pre"> PREV </v-btn>
      <v-btn
        v-if="this.targetInfo !== 'ondevice' && this.targetInfo !== ''"
        class="ma-0 pa-0"
        text
        style="color: #4a80ff"
        @click="next"
      >
        NEXT
      </v-btn>
      <v-btn v-else-if="target?.id" class="ma-0 pa-0" text style="color: #4a80ff" @click="create"> UPDATE </v-btn>
      <v-btn v-else class="ma-0 pa-0" text style="color: #4a80ff" @click="create"> CREATE </v-btn>
    </div>
  </div>
</template>
<script>
import { mapState } from "vuex";
import { TargetNamespace } from "@/store/modules/targetStore";
export default {
  data() {
    return {
      targetInfo: "",
      engine: "",
      os: "",
      osItem: [
        { value: "windows", label: "windows" },
        { value: "ubuntu", label: "ubuntu" },
        { value: "android", label: "android" }
      ],
      cpu: "",
      cpuItem: [
        { value: "arm", label: "ARM" },
        { value: "x86", label: "X86" }
      ],
      acc: "",
      accItem: [
        { value: "cuda", label: "cuda" },
        { value: "opencl", label: "opencl" },
        { value: "cpu", label: "cpu" }
      ],
      memory: 0
    };
  },

  computed: {
    ...mapState(TargetNamespace, ["target"])
  },

  watch: {
    targetInfo() {
      if (this.targetInfo === "ondevice") {
        this.$emit("isThridStep", false);
      } else {
        this.$emit("isThridStep", true);
      }
    }
  },

  mounted() {
    this.$emit("isThridStep", false);

    this.targetInfo = this.target?.info;
    this.engine = this.target?.engine;
    this.os = this.target?.os;
    this.cpu = this.target?.cpu;
    this.acc = this.target?.acc;
    this.memory = this.target?.memory;
  },

  methods: {
    pre() {
      this.$emit("prev");
    },
    next() {
      if (this.targetInfo === "" || !this.targetInfo) {
        this.$swal("Target", "Target Info를 입력해 주세요.", "error");
        return;
      } else if (this.engine === "" || !this.engine) {
        this.$swal("Target", "Target Engine을 입력해 주세요.", "error");
        return;
      } else if (this.os === "" || !this.os) {
        this.$swal("Target", "Target os를 입력해 주세요.", "error");
        return;
      } else if (this.acc === "" || !this.acc) {
        this.$swal("Target", "Target acc를 입력해 주세요.", "error");
        return;
      } else if (this.cpu === "" || !this.cpu) {
        this.$swal("Target", "Target cpu를 입력해 주세요.", "error");
        return;
      } else if (this.memory === "" || !this.memory) {
        this.$swal("Target", "Target memory를 입력해 주세요.", "error");
        return;
      }

      this.$emit("next", {
        info: this.targetInfo,
        engine: this.engine,
        os: this.os,
        acc: this.acc,
        cpu: this.cpu,
        memory: this.memory
      });
    },
    create() {
      this.$emit("create", {
        info: this.targetInfo,
        engine: this.engine,
        os: this.os,
        acc: this.acc,
        cpu: this.cpu,
        memory: this.memory
      });
    },
    osChange(value) {
      this.os = value;
    },
    cpuChange(value) {
      this.cpu = value;
    },
    accChange(value) {
      this.acc = value;
    },
    engineChange(value) {
      this.engine = value;
    },
    infoChange(value) {
      this.targetInfo = value;
    },
    memoryChange(value) {
      this.memory = value;
    }
  }
};
</script>
<style lang="scss" scoped></style>
