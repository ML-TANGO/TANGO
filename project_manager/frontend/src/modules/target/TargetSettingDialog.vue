<template lang="">
  <v-dialog v-model="dialog" persistent max-width="1100px" scrollable>
    <template v-slot:activator="{ on, attrs }">
      <div v-bind="attrs" v-on="on">
        <slot name="btn"></slot>
      </div>
    </template>
    <v-card>
      <v-card-title>
        <span v-if="target?.id" class="text-h5 font-weight-bold">Edit Target</span>
        <span v-else class="text-h5 font-weight-bold">Create Target</span>
        <v-card-actions>
          <v-btn color="blue darken-1" icon @click="close" absolute right>
            <v-icon color="#ccc" size="34">mdi-close-circle</v-icon>
          </v-btn>
        </v-card-actions>
      </v-card-title>
      <div class="d-flex align-center" style="height: 500px">
        <v-stepper v-model="e6" vertical class="elevation-0" style="width: 270px; letter-spacing: 1px" non-linear>
          <v-stepper-step :complete="e6 > 1" step="1"> Target Info <small>Enter Target Info</small> </v-stepper-step>
          <v-stepper-content step="1" class="my-3"> </v-stepper-content>
          <v-stepper-step :complete="e6 > 2" step="2">
            Target Specification <small>Select Specification</small>
          </v-stepper-step>
          <v-stepper-content step="2" class="my-3"></v-stepper-content>
          <!-- <v-stepper-step v-if="isThridStep" :complete="e6 > 3" step="3">
            Target Host <small>Select Host</small>
          </v-stepper-step>
          <v-stepper-content v-if="isThridStep" step="3" class="my-3"></v-stepper-content> -->

          <v-stepper-step :complete="e6 > 3" step="3"> Target Host <small>Select Host</small> </v-stepper-step>
          <v-stepper-content step="3" class="my-3"></v-stepper-content>
        </v-stepper>
        <div style="width: 75%; height: 500px" class="px-10">
          <FirstStepper v-if="e6 === 1" @next="next" />
          <!-- <SecondStepper
            v-else-if="e6 === 2"
            @next="next"
            @prev="prev"
            @isThridStep="setIsThridStep"
            @create="onCreate"
          /> -->
          <SecondStepper v-else-if="e6 === 2" @next="next" @prev="prev" @create="onCreate" />
          <ThirdStepper v-else-if="e6 === 3" @next="next" @prev="prev" @create="onCreate" />
        </div>
      </div>
    </v-card>
  </v-dialog>
</template>
<script>
import { mapMutations, mapState } from "vuex";
import { TargetNamespace, TargetMutations } from "@/store/modules/targetStore";

import FirstStepper from "./stepper/FirstStepper.vue";
import SecondStepper from "./stepper/SecondStepper.vue";
import ThirdStepper from "./stepper/ThirdStepper.vue";

import { createTarget, updateTarget } from "@/api";
export default {
  components: { FirstStepper, SecondStepper, ThirdStepper },

  data() {
    return {
      isThridStep: false,
      dialog: false,
      e6: 1
    };
  },

  computed: {
    ...mapState(TargetNamespace, ["target"])
  },

  watch: {
    dialog() {
      if (this.dialog) {
        this.e6 = 1;
      }
    }
  },

  mounted() {
    this.$EventBus.$on("previewRemove", async () => {
      this.SET_TARGET({ image: null });
    });
  },

  methods: {
    ...mapMutations(TargetNamespace, {
      SET_TARGET: TargetMutations.SET_TARGET
    }),

    next(data) {
      if (this.e6 !== 3) {
        this.e6 += 1;
      } else if (this.e6 !== 2) {
        this.e6 += 1;
      }

      this.SET_TARGET(data);
    },

    prev() {
      if (this.e6 !== 1) {
        this.e6 -= 1;
      }
    },

    setIsThridStep(data) {
      this.isThridStep = data;
    },

    async onCreate(data) {
      this.SET_TARGET(data);

      if (this.target?.id) {
        const params = {
          id: this.target.id,
          name: this.target.name,
          info: this.target.info,
          engine: this.target.engine,
          os: this.target.os,
          cpu: this.target.cpu,
          acc: this.target.acc,
          memory: this.target.memory,
          nfs_ip: this.target.nfs_ip || "",
          nfs_path: this.target.nfs_path || "",
          host_ip: this.target.host_ip,
          host_port: this.target.host_port,
          host_service_port: this.target.host_service_port,
          image: this.target.image || ""
        };
        await updateTarget(params);
      } else {
        await createTarget(this.target);
      }

      this.dialog = false;
      this.$emit("close");
    },

    close() {
      this.dialog = false;
      this.$emit("close");

      this.e6 = 0;
    }
  }
};
</script>
<style lang="scss" scoped></style>
