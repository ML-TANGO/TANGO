<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <div class="d-flex justify-space-between" style="width: 100%">
        <p class="text-h5 mb-4" style="color: #4a80ff">Target</p>
        <p class="mr-4" v-if="selected !== -1">
          Selected Target: <span style="color: #4a80ff"> {{ items[selected]?.name }} </span>
        </p>
      </div>
      <div style="width: 100%; height: 350px; backface-visibility: hidden; flex: 1 1 auto; overflow-y: auto">
        <div v-for="(item, index) in items" :key="index">
          <TargetCard
            :isSelected="selected === index"
            :isHover="hoverIndex === index"
            :item="item"
            @click="onSelected(item, index)"
            @mouseover="onMouseover(index)"
            @mouseleave="onMouseleave"
          >
          </TargetCard>
        </div>
      </div>
    </div>
    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="pre"> PREV </v-btn>
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="next"> NEXT </v-btn>
    </div>
  </div>
</template>
<script>
import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import TargetCard from "@/modules/common/card/TargetCard.vue";

import { getTargetList } from "@/api";

export default {
  components: { TargetCard },
  data() {
    return {
      hoverIndex: -1,
      selected: -1,
      selelctedItem: null,
      items: []
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["selectedTarget"])
  },

  async created() {
    try {
      this.items = await getTargetList();

      if (this.selectedTarget?.id) {
        this.selected = this.items.findIndex(q => q.id === this.selectedTarget.id);
        this.selelctedItem = this.selectedTarget;
      }
    } catch {
      this.items = [];
    }
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_SELECTED_TARGET: ProjectMutations.SET_SELECTED_TARGET
    }),

    pre() {
      this.$emit("prev");
    },

    next() {
      if (!this.selelctedItem) return;
      this.$emit("next", { target_id: this.selelctedItem.id });
      this.SET_SELECTED_TARGET(this.selelctedItem);
    },

    onMouseover(index) {
      this.hoverIndex = index;
    },

    onMouseleave() {
      this.hoverIndex = -1;
    },

    onSelected(item, index) {
      // TODO index 대신 item id로 변경
      this.selected = index;

      this.selelctedItem = item;
    }
  }
};
</script>
<style lang="scss" scoped></style>
