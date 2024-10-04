<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%; width: 100%">
    <div style="width: 100%">
      <div class="d-flex justify-space-between" style="width: 100%">
        <p class="text-h5 mb-4" style="color: #4a80ff">DataSet</p>
        <p class="mr-4" v-if="selected !== -1">
          Selected Dataset: <span style="color: #4a80ff"> {{ selectedItem.name }} </span>
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
import Vue from "vue";

import { mapMutations, mapState } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import DatasetCard from "@/modules/common/card/DatasetCard.vue";

import { DatasetStatus, CommonDatasetName, LearningType } from "@/shared/enums";

import { getDatasetListTango, getDatasetFolderSize, getDatasetFileCount } from "@/api";

export default {
  components: { DatasetCard },

  data() {
    return {
      hoverIndex: -1,
      selected: -1,
      selectedItem: null,
      items: []
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["selectedImage", "project"])
  },

  async created() {
    try {
      this.items = await getDatasetListTango();
      this.items = this.items.filter(q => q.name !== "tmp");
      this.items = this.items.filter(q => q.status === DatasetStatus.COMPLETE);
      this.getDatasetSize();
      this.getDatasetCount();

      if (this.selectedImage?.name) {
        this.selected = this.items.findIndex(q => q.name === this.selectedImage.name);
        this.selectedItem = this.selectedImage;
      }
    } catch {
      this.items = [];
    }
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      SET_SELECTED_IMAGE: ProjectMutations.SET_SELECTED_IMAGE
    }),

    getDatasetSize() {
      const folderList = this.items.map(q => q.path);
      getDatasetFolderSize(folderList).then(res => {
        res.datas.forEach(data => {
          const target = this.items.find(q => q.path === data.folder_path);
          // target.size = data.size;

          Vue.set(target, "size", data.size);
        });

        // this.listKey++;
      });
    },

    getDatasetCount() {
      const folderList = this.items.map(q => q.path);
      getDatasetFileCount(folderList).then(res => {
        res.datas.forEach(data => {
          const target = this.items.find(q => q.path === data.folder_path);
          // target.file_count = data.count;
          Vue.set(target, "file_count", data.count);
        });

        // this.listKey++;
      });
    },

    pre() {
      this.$emit("prev");
    },

    next() {
      if (!this.selectedItem) return;

      let learning_type = this.project.learning_type;
      if (this.selectedItem.name !== CommonDatasetName.COCO && learning_type === LearningType.INCREMENTAL) {
        learning_type = LearningType.NORMAL;
      }

      this.$emit("next", { dataset: this.selectedItem.name, learning_type });
      this.SET_SELECTED_IMAGE(this.selectedItem);
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

      this.selectedItem = item;
    }
  }
};
</script>
<style lang="scss" scoped></style>
