<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <div class="d-flex justify-space-between" style="width: 100%">
        <p class="text-h5 mb-4" style="color: #4a80ff">DataSet</p>
        <p class="mr-4" v-if="selected !== -1">
          Selected Dataset: <span style="color: #4a80ff"> {{ selelctedItem.TITLE }} </span>
        </p>
      </div>
      <div style="width: 100%; height: 350px; backface-visibility: hidden; flex: 1 1 auto; overflow-y: auto">
        <!--  class="mb-3 mr-2 d-flex pa-3 align-center" -->
        <div v-for="(item, index) in items" :key="index">
          <DatasetCard
            :item="item"
            :isSelected="selected === index"
            :isHover="hoverIndex === index"
            @click="onSelected(item, index)"
            @mouseover="onMouseover(index)"
            @mouseleave="onMouseleave"
          >
          </DatasetCard>
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

import DatasetCard from "@/modules/common/card/DatasetCard.vue";

import { getDatasetList } from "@/api";

export default {
  components: { DatasetCard },

  data() {
    return {
      hoverIndex: -1,
      selected: -1,
      selelctedItem: null,
      items: []
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["selectedImage"])
  },

  async created() {
    try {
      this.items = await getDatasetList();

      if (this.selectedImage?.DATASET_CD) {
        this.selected = this.items.findIndex(q => q.DATASET_CD === this.selectedImage.DATASET_CD);
        this.selelctedItem = this.selectedImage;
      }
    } catch {
      console.log("dataset error");
      this.items = [];
    }
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_SELECTED_IMAGE: ProjectMutations.SET_SELECTED_IMAGE
    }),

    pre() {
      this.$emit("prev");
    },

    next() {
      console.log("this.selelctedItem", this.selelctedItem);
      this.$emit("next", { dataset: this.selelctedItem.DATASET_CD });
      this.SET_SELECTED_IMAGE(this.selelctedItem);
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
