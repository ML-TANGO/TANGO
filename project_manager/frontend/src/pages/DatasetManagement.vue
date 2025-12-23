<template>
  <div style="height: 100%">
    <v-card style="height: 100%; padding-right: 22px" class="py-5 pl-10">
      <div style="max-height: calc(100vh - 90px); overflow: auto" class="mt-1 grid-view">
        <div v-for="(item, index) in items" :key="index" class="d-flex">
          <DatasetCard :item="item" isHover="true" style="width: 95%" @click="openPreview(item)" />
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
    <v-dialog v-model="previewDialog" max-width="1200">
      <v-card>
        <v-card-title class="d-flex align-center justify-space-between">
          <div class="d-flex flex-column">
            <span>Dataset Preview</span>
            <small style="color: #888; font-size: 12px">{{ previewDatasetName }}</small>
          </div>
          <v-btn icon @click="previewDialog = false">
            <v-icon>mdi-close</v-icon>
          </v-btn>
        </v-card-title>
        <v-card-text>
          <v-progress-linear v-if="previewLoading" indeterminate color="tango" class="mb-4" />
          <v-alert v-else-if="previewError" type="error" dense text>{{ previewError }}</v-alert>
          <div v-else class="preview-grid">
            <div v-if="previewSamples.length === 0" class="preview-empty">표시할 샘플이 없습니다.</div>
            <div v-for="(sample, idx) in previewSamples" :key="idx" class="preview-item">
              <v-img :src="sample.image" max-height="260" contain></v-img>
              <div class="mt-2">
                <div class="preview-file">{{ sample.file }}</div>
                <div v-if="sample.legend?.length" class="preview-legend">
                  <span
                    v-for="(legend, legendIdx) in sample.legend"
                    :key="legendIdx"
                    class="preview-legend-item"
                    :style="legendStyle(legend)"
                  >
                    {{ legend.abbr }}: {{ legend.label }}
                  </span>
                </div>
                <div v-else-if="sample.label_items?.length" class="preview-labels">
                  <span
                    v-for="(labelItem, labelIdx) in sample.label_items"
                    :key="labelIdx"
                    class="preview-label-item"
                    :style="labelItemStyle(labelItem)"
                  >
                    {{ labelItem.text }}
                  </span>
                </div>
                <div v-else-if="sample.labels?.length" class="preview-labels">
                  {{ sample.labels.join(", ") }}
                </div>
              </div>
            </div>
          </div>
        </v-card-text>
      </v-card>
    </v-dialog>
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
  getDatasetFileCount,
  getDatasetPreview
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
      interval: null,
      previewDialog: false,
      previewLoading: false,
      previewError: "",
      previewSamples: [],
      previewDatasetName: ""
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

    async openPreview(datasetInfo) {
      this.previewDialog = true;
      this.previewLoading = true;
      this.previewError = "";
      this.previewSamples = [];
      this.previewDatasetName = datasetInfo?.name || "";
      try {
        const res = await getDatasetPreview(datasetInfo?.name, 5);
        if (res?.status !== 200) {
          this.previewError = res?.message || "미리보기 로딩에 실패했습니다.";
          return;
        }
        this.previewSamples = res?.samples || [];
      } catch (error) {
        this.previewError = "미리보기 로딩에 실패했습니다.";
      } finally {
        this.previewLoading = false;
      }
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
    },

    legendStyle(legend) {
      const isPerson = legend?.label?.toLowerCase?.() === "person";
      const color = isPerson ? "#000" : this.legendColor(legend?.color_hex || legend?.color);
      const extra = this.nearWhite(color) ? { textShadow: "0 0 2px #000" } : {};
      return {
        color: color,
        borderColor: color,
        ...extra
      };
    },

    labelItemStyle(labelItem) {
      const color = this.legendColor(labelItem?.color_hex || labelItem?.color);
      const extra = this.nearWhite(color) ? { textShadow: "0 0 2px #000" } : {};
      return { color: color, ...extra };
    },

    legendColor(color) {
      if (!color) return "#444";
      if (typeof color === "string") {
        return color;
      }
      if (Array.isArray(color) && color.length === 3) {
        return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
      }
      return "#444";
    },

    nearWhite(color) {
      if (!color) return false;
      if (typeof color === "string" && color.startsWith("#") && color.length === 7) {
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);
        return r > 220 && g > 220 && b > 220;
      }
      if (typeof color === "string" && color.startsWith("rgb")) {
        const nums = color.replace(/[^\d,]/g, "").split(",").map(val => parseInt(val.trim(), 10));
        if (nums.length >= 3) return nums[0] > 220 && nums[1] > 220 && nums[2] > 220;
      }
      return false;
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

.preview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}

.preview-item {
  background: #f7f7f7;
  border-radius: 8px;
  padding: 12px;
}

.preview-empty {
  grid-column: 1 / -1;
  text-align: center;
  color: #777;
  padding: 40px 0;
}

.preview-file {
  color: #4a80ff;
  font-weight: bold;
  font-size: 13px;
}

.preview-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 12px;
  margin-top: 6px;
  font-size: 16px;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica,
    Arial, sans-serif;
}

.preview-legend-item {
  font-weight: 600;
  border-left: 3px solid transparent;
  padding-left: 6px;
}

.preview-labels {
  margin-top: 6px;
  font-size: 18px;
  font-weight: 600;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Helvetica,
    Arial, sans-serif;
  color: #222;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.preview-label-item {
  border-left: 3px solid transparent;
  padding-left: 6px;
}

@media (max-width: 900px) {
  .preview-grid {
    grid-template-columns: 1fr;
  }
}
</style>
