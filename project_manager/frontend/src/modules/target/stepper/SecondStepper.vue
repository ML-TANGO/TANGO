<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <p class="text-h5 mb-4" style="color: #4a80ff">Target Specification</p>

      <!-- Target Info -->
      <div class="d-flex align-center mt-11" style="gap: 25px">
        <div style="width: 150px">Target Info</div>
        <!-- <v-radio-group :value="targetInfo" row hide-details="" class="ma-0" @change="infoChange">
          <v-radio label="cloud" value="cloud"></v-radio>
          <v-radio label="kubernetes" value="k8s"></v-radio>
          <v-radio label="PC-Server" value="PCServer"></v-radio>
          <v-radio label="PC or ondevice" value="ondevice"></v-radio>
        </v-radio-group> -->

        <v-autocomplete
          :value="targetInfo"
          :items="TargetInfoList"
          label="Target Info"
          outlined
          dense
          hide-details
          item-text="value"
          item-value="key"
          @change="infoChange"
        />
      </div>

      <v-divider class="mt-3 mb-3"></v-divider>

      <!-- Engine -->
      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">Engine</div>
        <v-radio-group :value="engine" row hide-details="" class="ma-0" @change="engineChange" :disabled="!targetInfo">
          <v-radio
            v-for="(item, index) in allowedEngine"
            :key="index"
            :label="item.label"
            :value="item.value"
          ></v-radio>
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
              label="Memory (MB)"
              hide-details
              style="width: 50%"
              @change="memoryChange"
            />
          </div>
        </div>
      </div>

      <v-divider class="mt-3 mb-3" v-if="selectedTargetInfo?.value.toLowerCase().includes('k8s')"></v-divider>

      <div
        class="d-flex mb-2 align-center"
        style="gap: 25px"
        v-if="selectedTargetInfo?.value.toLowerCase().includes('k8s')"
      >
        <div style="min-width: 150px">nfs IP</div>
        <v-text-field :value="nfsIP" outlined dense label="nfs ip" hide-details @change="nfsIPChange" />
      </div>

      <div class="d-flex align-center" style="gap: 25px" v-if="selectedTargetInfo?.value.toLowerCase().includes('k8s')">
        <div style="min-width: 150px">nfs Path</div>
        <v-text-field :value="nfsPath" outlined dense label="nfs path" hide-details @change="nfsPathChange" />
      </div>
    </div>
    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="pre"> PREV </v-btn>
      <!-- <v-btn
        v-if="this.targetInfo !== 'ondevice' && this.targetInfo !== ''"
        class="ma-0 pa-0"
        text
        style="color: #4a80ff"
        @click="next"
      >
        NEXT
      </v-btn> -->
      <!-- <v-btn 
      v-else-if="target?.id" class="ma-0 pa-0" text style="color: #4a80ff" @click="create"> UPDATE </v-btn> -->
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="next"> NEXT </v-btn>
      <!-- <v-btn v-else class="ma-0 pa-0" text style="color: #4a80ff" @click="create"> CREATE </v-btn> -->
    </div>
  </div>
</template>
<script>
import { mapState } from "vuex";
import { TargetNamespace } from "@/store/modules/targetStore";

import { TargetInfoList, EngineLabel } from "@/shared/enums";
export default {
  data() {
    return {
      TargetInfoList,
      EngineLabel,
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
      memory: 0,
      nfsIP: "",
      nfsPath: ""
    };
  },

  computed: {
    ...mapState(TargetNamespace, ["target"]),

    selectedTargetInfo() {
      return TargetInfoList.find(q => q.value === this.targetInfo);
    },

    allowedEngine() {
      if (!this.selectedTargetInfo) {
        return Object.entries(EngineLabel).map(([value, label]) => ({ label: label, value: value }));
      }
      return this.selectedTargetInfo.allowedEngine.map(q => ({ label: EngineLabel[q], value: q }));
    }
  },

  watch: {
    targetInfo() {
      //   if (this.targetInfo === "ondevice") {
      //     this.$emit("isThridStep", false);
      //   } else {
      //     this.$emit("isThridStep", true);
      //   }
    }
  },

  mounted() {
    // this.$emit("isThridStep", false);

    this.targetInfo = this.target?.info;
    this.engine = this.target?.engine;
    this.os = this.target?.os;
    this.cpu = this.target?.cpu;
    this.acc = this.target?.acc;
    this.memory = this.target?.memory;

    this.nfsIP = this.target?.nfs_ip;
    this.nfsPath = this.target?.nfs_path;
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

      if (this.selectedTargetInfo?.value.toLowerCase().includes("k8s") && (this.nfsIP === "" || !this.nfsIP)) {
        this.$swal("Target", "Target nfs ip 입력해 주세요.", "error");
        return;
      }
      if (this.selectedTargetInfo?.value.toLowerCase().includes("k8s") && (this.nfsPath === "" || !this.nfsPath)) {
        this.$swal("Target", "Target nfs path 입력해 주세요.", "error");
        return;
      }

      this.$emit("next", {
        info: this.targetInfo,
        engine: this.engine,
        os: this.os,
        acc: this.acc,
        cpu: this.cpu,
        memory: this.memory,
        nfs_ip: this.nfsIP || "",
        nfs_path: this.nfsPath || ""
      });
    },
    create() {
      this.$emit("create", {
        info: this.targetInfo,
        engine: this.engine,
        os: this.os,
        acc: this.acc,
        cpu: this.cpu,
        memory: this.memory,
        nfs_ip: this.nfsIP || "",
        nfs_path: this.nfsPath || ""
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
      this.engine = null;
    },
    memoryChange(value) {
      this.memory = value;
    },
    nfsIPChange(value) {
      this.nfsIP = value;
    },
    nfsPathChange(value) {
      this.nfsPath = value;
    }
  }
};
</script>
<style lang="scss" scoped></style>
