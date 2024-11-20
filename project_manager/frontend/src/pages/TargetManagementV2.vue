<template>
  <v-card style="height: 100%; padding-right: 22px" class="py-5 pl-10">
    <div style="height: calc(95vh - 90px); overflow: auto" class="mt-8">
      <div
        v-for="(item, index) in items"
        :key="index"
        style="position: relative"
        @mouseover="onMouseover(index)"
        @mouseleave="onMouseleave"
      >
        <div style="width: calc(100% - 100px); height: 100%">
          <TargetCard ref="targetCardRef" :item="item" :isHover="hoverIndex === index" />
        </div>
        <div
          :style="{ backgroundColor: hoverIndex === index ? '#EFEFEF' : '#f7f7f7' }"
          class="target-card-wrapper d-flex align-center justify-center"
        >
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
        <v-btn color="tango" dark absolute style="top: 4px; right: 30px" height="40" width="180">
          NEW TARGET&nbsp;<v-icon :size="20">mdi-plus</v-icon>
        </v-btn>
      </template>
    </TargetSettingDialog>
  </v-card>
</template>
<script>
import Swal from "sweetalert2";
import { mapMutations } from "vuex";
import { TargetNamespace, TargetMutations } from "@/store/modules/targetStore";

import Tango_logo from "@/assets/icon_3x/Tango_logo.png";
import TargetSettingDialog from "@/modules/target/TargetSettingDialog.vue";

import { getTargetList, deleteTarget } from "@/api";

import prettyBytes from "pretty-bytes";

import TargetCard from "@/modules/common/card/TargetCardV2.vue";

export default {
  components: { TargetSettingDialog, TargetCard },
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

  mounted() {
    window.addEventListener("resize", this.onResize);
  },

  destroyed() {
    window.removeEventListener("resize", this.onResize);
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
        Swal.fire({
          title: `${target.name} Target을 \n 삭제하시겠습니까?`,
          text: "삭제한 뒤 복구가 불가능합니다.",
          icon: "warning",
          showCancelButton: true,
          confirmButtonColor: "#3085d6",
          cancelButtonColor: "#d33",
          confirmButtonText: "확인",
          cancelButtonText: "취소"
        }).then(async result => {
          if (result.isConfirmed) {
            await deleteTarget(target.id)
              .then(async () => {
                this.items = await getTargetList();
              })
              .catch(() => {
                Swal.fire("사용중인 타겟입니다.", "", "error");
              });
          }
        });
      } catch (err) {
        Swal.fire("사용중인 타겟입니다.", "", "error");
      }
    },

    onEdit(target) {
      this.SET_TARGET(target);
    },

    onResize() {
      const targetCardRefs = this.$refs.targetCardRef;

      if (Array.isArray(targetCardRefs)) {
        Array.from(targetCardRefs).forEach(targetCardRef => {
          targetCardRef.setAnimation();
        });
      } else {
        targetCardRefs.setAnimation();
      }
    }
  }
};
</script>
<style lang="css" scoped>
.target-card-wrapper {
  position: absolute;
  z-index: 99;
  right: 0px;
  top: 0px;
  width: 125px;
  height: 100%;
  border-radius: 4px;
}
</style>
