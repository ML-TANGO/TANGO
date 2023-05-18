<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <p class="text-h5 mb-4" style="color: #4a80ff">Target Host</p>

      <!-- IP Address -->
      <div class="d-flex align-center mb-5 mt-11" style="gap: 25px">
        <div style="width: 150px">IP Address</div>
        <v-text-field :value="ipAddress" outlined dense label="IP Address" hide-details @change="ipChange" />
      </div>

      <!-- Port -->
      <div class="d-flex align-center mb-5" style="gap: 25px">
        <div style="width: 150px">Port</div>
        <v-text-field :value="port" outlined dense label="Port" hide-details @change="portChange" />
      </div>

      <!-- Service Port -->
      <div class="d-flex align-center" style="gap: 25px">
        <div style="width: 150px">Service Port</div>
        <v-text-field
          :value="servicePort"
          outlined
          dense
          label="Service Port"
          hide-details
          @change="servicePortChange"
        />
      </div>
    </div>

    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="pre"> PREV </v-btn>
      <v-btn v-if="target?.id" class="ma-0 pa-0" text style="color: #4a80ff" @click="create"> UPDATE </v-btn>
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
      ipAddress: "",
      port: "",
      servicePort: ""
    };
  },

  computed: {
    ...mapState(TargetNamespace, ["target"])
  },

  mounted() {
    this.ipAddress = this.target?.host_ip || "";
    this.port = this.target?.host_port || "";
    this.servicePort = this.target?.host_service_port || "";
  },

  methods: {
    pre() {
      this.$emit("prev");
    },

    create() {
      if (this.ipAddress === "" || !this.ipAddress) {
        this.$swal("Target", "Ip Address를 입력해 주세요.", "error");
        return;
      } else if (this.port === "" || !this.port) {
        this.$swal("Target", "Port를 입력해 주세요.", "error");
        return;
      } else if (this.servicePort === "" || !this.servicePort) {
        this.$swal("Target", "Service Port를 입력해 주세요.", "error");
        return;
      }

      this.$emit("create", { host_ip: this.ipAddress, host_port: this.port, host_service_port: this.servicePort });
    },

    ipChange(value) {
      this.ipAddress = value;
    },

    portChange(value) {
      this.port = value;
    },

    servicePortChange(value) {
      this.servicePort = value;
    }
  }
};
</script>
<style lang="scss" scoped></style>
