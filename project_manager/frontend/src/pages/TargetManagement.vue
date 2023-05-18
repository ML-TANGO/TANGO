<template>
  <v-card style="height: 100%" class="py-5 px-10">
    <div style="height: calc(100vh - 90px); overflow: auto">
      <div
        v-for="(item, index) in items"
        :key="index"
        class="mb-3 mr-2 d-flex pa-3 align-center"
        style="border-radius: 4px"
        :style="{ background: hoverIndex === index ? '#DFDFDF' : '#f1f1f1' }"
        @mouseover="onMouseover(index)"
        @mouseleave="onMouseleave"
      >
        <div class="d-flex justify-center align-center pa-3">
          <v-img :src="item.image" width="50" max-width="50" contain></v-img>
        </div>
        <div class="ml-3 d-flex flex-column justify-center" style="width: 400px">
          <div class="d-flex align-center" style="gap: 10px">
            <p style="color: #000000ff; letter-spacing: 1px; font-size: 14px" class="pa-0 ma-0">{{ item.name }}</p>
          </div>
          <div class="d-flex align-center mt-3" style="gap: 10px">
            <small style="color: #aaa; font-size: 11px; min-width: 70px">{{ item.create_date.slice(0, 10) }}</small>
            <small style="color: #aaa; font-size: 11px; min-width: 95px">
              created by
              <span style="color: #4a80ff">{{ item.create_user }}</span>
            </small>
          </div>
        </div>
        <v-divider vertical class="ml-5 mr-5" />
        <div style="width: 100%; height: 100%" class="ml-2">
          <div class="d-flex align-center" style="gap: 35px; height: 100%">
            <div class="d-flex flex-column text-center" style="width: 80px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Info</small>
              <p class="ma-0" style="font-size: 13px">{{ item.info }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 80px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Memory</small>
              <p class="ma-0" style="font-size: 13px">{{ prettyBytes(Number(item?.memory) || 0) }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 80px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">OS</small>
              <p class="ma-0" style="font-size: 13px">{{ item.os }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 60px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">CPU</small>
              <p class="ma-0" style="font-size: 13px">{{ item.cpu }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 80px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Accelerator</small>
              <p class="ma-0" style="font-size: 13px">{{ item.acc }}</p>
            </div>
            <div class="d-flex flex-column text-center" style="width: 60px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Engine</small>
              <p class="ma-0" style="font-size: 13px">{{ item.engine }}</p>
            </div>
            <div v-if="item.host_ip !== ''" class="d-flex flex-column text-center" style="width: 80px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">IP Address</small>
              <p class="ma-0" style="font-size: 13px">{{ item.host_ip }}</p>
            </div>
            <div v-if="item.host_port !== ''" class="d-flex flex-column text-center" style="width: 80px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Port</small>
              <p class="ma-0" style="font-size: 13px">{{ item.host_port }}</p>
            </div>
            <div v-if="item.host_service_port !== ''" class="d-flex flex-column text-center" style="width: 80px">
              <small class="mb-2" style="letter-spacing: 1px; color: #aaa; font-size: 10px">Service Port</small>
              <p class="ma-0" style="font-size: 13px">{{ item.host_service_port }}</p>
            </div>
          </div>
        </div>
        <div class="d-flex">
          <TargetSettingDialog @close="onClose">
            <template v-slot:btn>
              <v-btn icon @click="onEdit(item)">
                <v-icon>mdi-pencil-outline</v-icon>
              </v-btn>
            </template>
          </TargetSettingDialog>
          <v-btn icon @click="onDelete(item)">
            <v-icon>mdi-delete</v-icon>
          </v-btn>
        </div>
      </div>
    </div>
    <TargetSettingDialog @close="onClose">
      <template v-slot:btn>
        <v-btn
          class="elevation-23"
          color="tango"
          dark
          absolute
          bottom
          right
          style="bottom: 67px; right: 60px"
          height="50"
          width="180"
          rounded
        >
          NEW TARGET&nbsp;<v-icon :size="20">mdi-plus</v-icon>
        </v-btn>
      </template>
    </TargetSettingDialog>
  </v-card>
</template>
<script>
import { mapMutations } from "vuex";
import { TargetNamespace, TargetMutations } from "@/store/modules/targetStore";

import Tango_logo from "@/assets/icon_3x/Tango_logo.png";
import TargetSettingDialog from "@/modules/target/TargetSettingDialog.vue";

import { getTargetList, deleteTarget } from "@/api";

import prettyBytes from "pretty-bytes";

export default {
  components: { TargetSettingDialog },
  data() {
    return {
      Tango_logo,
      items: [],
      selected: -1,
      hoverIndex: -1,
      prettyBytes
    };
  },

  async created() {
    this.items = await getTargetList();
  },

  methods: {
    ...mapMutations(TargetNamespace, {
      INIT_TARGET: TargetMutations.INIT_TARGET,
      SET_TARGET: TargetMutations.SET_TARGET
    }),

    onMouseover(index) {
      this.hoverIndex = index;
    },

    onMouseleave() {
      this.hoverIndex = -1;
    },

    async onClose() {
      this.INIT_TARGET();
      this.items = await getTargetList();
    },

    async onDelete(target) {
      try {
        this.$swal
          .fire({
            title: `${target.name} Target을 \n 삭제하시겠습니까?`,
            text: "삭제한 뒤 복구가 불가능합니다.",
            icon: "warning",
            showCancelButton: true,
            confirmButtonColor: "#3085d6",
            cancelButtonColor: "#d33",
            confirmButtonText: "확인",
            cancelButtonText: "취소"
          })
          .then(async result => {
            if (result.isConfirmed) {
              await deleteTarget(target.id)
                .then(async () => {
                  this.items = await getTargetList();
                })
                .catch(() => {
                  this.$swal("사용중인 타겟입니다.", "", "error");
                });
            }
          });
      } catch (err) {
        this.$swal("사용중인 타겟입니다.", "", "error");
      }
    },

    onEdit(target) {
      this.SET_TARGET(target);
    }
  }
};
</script>
<style lang="scss"></style>
