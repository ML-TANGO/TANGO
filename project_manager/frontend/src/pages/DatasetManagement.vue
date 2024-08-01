<template>
  <div style="height: 100%">
    <v-card style="height: 100%; padding-right: 22px" class="py-5 pl-10">
      <div style="max-height: calc(100vh - 90px); overflow: auto" class="mt-1 grid-view">
        <div v-for="(item, index) in items" :key="index" class="d-flex">
          <DatasetCard :item="item" isHover="true" style="width: 95%" />
          <div
            class="d-flex justify-center align-center my-3 pr-2"
            style="background-color: #efefef; margin-left: -15px; border-radius: 0px 4px 4px 0px"
          >
            <v-btn icon v-if="item.status === DatasetStatus.NONE" @click="datasetDownload(item)">
              <v-icon>mdi-download</v-icon>
            </v-btn>
            <v-progress-circular v-else-if="item.status === DatasetStatus.DOWNLOADING" indeterminate color="tango" />
            <v-icon v-else class="pr-1"> mdi-check-underline-circle</v-icon>
          </div>
        </div>
      </div>
    </v-card>
    <KaggleUserInfoDialog ref="KaggleUserInfoDialogref" @restart="kaggleDownload"></KaggleUserInfoDialog>
  </div>
</template>
<script>
import Vue from "vue";

import Tango_logo from "@/assets/icon_3x/Tango_logo.png";
import KaggleUserInfoDialog from "@/modules/common/dialog/KaggleUserInfoDialog.vue";

import {
  getDatasetListTango,
  checkExistKaggleInfo,
  imagenetDatasetDownload,
  cocoDatasetDownload,
  vocDatasetDownload,
  chestXrayDatasetDownload,
  getDatasetFolderSize,
  getDatasetFileCount
} from "@/api";

import prettyBytes from "pretty-bytes";
import DatasetCard from "@/modules/common/card/DatasetCard.vue";

import { CommonDatasetName, DatasetStatus } from "@/shared/enums";

export default {
  components: { DatasetCard, KaggleUserInfoDialog },
  data() {
    return {
      Tango_logo,
      items: [],
      selected: -1,
      hoverIndex: -1,
      prettyBytes,
      DatasetStatus,
      interval: null
    };
  },

  async mounted() {
    await this.loadDatasets();
    if (this.items.some(item => item.status === DatasetStatus.DOWNLOADING)) {
      this.startInterval();
    }
  },

  destroyed() {
    this.stopInterval();
  },

  methods: {
    async loadDatasets() {
      this.items = await getDatasetListTango();
      this.items = this.items?.filter(q => q.name !== "tmp") || [];
      this.getDatasetSize();
      this.getDatasetCount();
    },

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

    startInterval() {
      this.loadDatasets();

      if (this.interval) return;

      const time = 5 * 60 * 1000; //5 min
      this.interval = setInterval(() => {
        this.loadDatasets();
      }, time);
    },

    stopInterval() {
      if (this.interval) {
        clearInterval(this.interval);
      }
    },

    onMouseover(index) {
      this.hoverIndex = index;
    },

    onMouseleave() {
      this.hoverIndex = -1;
    },

    async datasetDownload(datasetInfo) {
      if (datasetInfo.name === CommonDatasetName.IMAGE_NET) {
        await imagenetDatasetDownload();
        this.startInterval();
      } else if (datasetInfo.name === CommonDatasetName.CHESTXRAY) {
        const checkKaggleInfo = await checkExistKaggleInfo();
        if (!checkKaggleInfo.isExist) {
          this.$refs.KaggleUserInfoDialogref.isOpen = true;
          return;
        }
        await chestXrayDatasetDownload();
        this.startInterval();
      } else if (datasetInfo.name === CommonDatasetName.VOC) {
        await vocDatasetDownload();
        this.startInterval();
      } else if (datasetInfo.name === CommonDatasetName.COCO) {
        await cocoDatasetDownload();
        this.startInterval();
      }
    },

    kaggleDownload() {
      chestXrayDatasetDownload();
      this.startInterval();
    }
  }
};
</script>
<style lang="css" scoped>
.grid-view {
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 15px;
}
</style>
