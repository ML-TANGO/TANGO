<template>
  <div class="train-grid">
    <div class="d-flex align-end" style="width: 100%">
      <div
        class="px-3 py-1 d-flex align-center justify-center"
        style="border-radius: 100px; background-color: #ef3e2b; color: white; width: 100%"
      >
        TRAIN
      </div>
    </div>
    <div style="grid-row: 1/2; grid-column: 2/5">
      <TrainContainer primaryColor="#f79f95" secondaryColor="#f36f60" :data="trainLossLog" />
    </div>
    <div style="grid-row: 1/3; grid-column: 5/6">
      <div class="text-center">
        <v-progress-circular :rotate="-90" :size="160" :width="10" :value="epochProgress" color="#625e63">
          EPOCH #{{ train?.epoch || 0 }}
          <!-- <br /> -->
          <!-- {{ train?.epoch || 0 }} / {{ train?.total_epoch || 0 }} -->
        </v-progress-circular>
      </div>
    </div>
    <div class="d-flex align-end" style="width: 100%">
      <div
        class="px-3 py-1 d-flex align-center justify-center"
        style="border-radius: 100px; background-color: #34879f; color: white; width: 100%"
      >
        VAL
      </div>
    </div>
    <div style="grid-row: 2/3; grid-column: 2/5; height: 90px">
      <div style="width: 100%; height: 100%" class="d-flex align-end">
        <TrainContainer primaryColor="#9ac3cf" secondaryColor="#67a5b7" :data="valAccuracyLog" />
      </div>
    </div>
    <div style="grid-row: 3/4; grid-column: 2/3" class="d-flex justify-center">
      <ChartContainer primaryColor="#ef3e2b" title="TRAINING LOSS">
        <template #chart-area>
          <div class="d-flex justify-center">
            <div
              class="d-flex justify-center"
              style="width: 100%; height: 100%; border: 1px solid #ef3e2b; border-radius: 8px; padding-top: 25px"
            >
              <div style="width: 300px; height: 280px">
                <LineChart xTitle="epoch" yTitle="Total loss" :chartData="lossChartData" />
              </div>
            </div>
          </div>
        </template>
      </ChartContainer>
    </div>
    <div style="grid-row: 3/4; grid-column: 3/5" class="d-flex justify-center">
      <ChartContainer primaryColor="#006fc0" title="VALIDATION ACCURACY">
        <template #chart-area>
          <div class="d-flex justify-center" v-if="taskType === TaskType.DETECTION">
            <div
              class="d-flex justify-center"
              style="width: 100%; height: 100%; border: 1px solid #006fc0; border-radius: 8px; padding-top: 25px"
            >
              <div style="width: 300px; height: 280px">
                <LineChart xTitle="epoch" yTitle="mAP@0.5" :chartData="mAP50ChartData" />
              </div>
            </div>
            <div style="width: 20px"></div>
            <div
              class="d-flex justify-center"
              style="width: 100%; height: 100%; border: 1px solid #006fc0; border-radius: 8px; padding-top: 25px"
            >
              <div style="width: 300px; height: 280px">
                <LineChart xTitle="epoch" yTitle="mAP" :chartData="mAPChartData" />
              </div>
            </div>
          </div>
          <!-- ----------------------------------------------------------------------------- -->
          <div v-else>
            <div
              class="d-flex justify-center"
              style="width: 100%; height: 100%; border: 1px solid #006fc0; border-radius: 8px; padding-top: 25px"
            >
              <div style="width: 300px; height: 280px">
                <LineChart xTitle="epoch" yTitle="Top-1 Accuracy" :chartData="classificationChartData" />
              </div>
            </div>
          </div>
        </template>
      </ChartContainer>
    </div>
    <div style="grid-row: 3/4; grid-column: 5/6" class="d-flex justify-center">
      <ChartContainer primaryColor="#949494" title="ELAPSED TIME">
        <template #chart-area>
          <div class="d-flex justify-center">
            <div
              class="d-flex justify-center"
              style="width: 100%; height: 100%; border: 1px solid #949494; border-radius: 8px; padding-top: 25px"
            >
              <div style="width: 300px; height: 280px">
                <BarLineMixinChart xTitle="epoch" yTitle="Time (s)" :chartData="elapsedTimeChartData" />
              </div>
            </div>
          </div>
        </template>
      </ChartContainer>
    </div>
  </div>
</template>
<script>
import { mapState } from "vuex";
import { ProjectNamespace } from "@/store/modules/project";

import TrainContainer from "../components/train/TrainContainer.vue";
import ChartContainer from "../components/train/ChartContainer.vue";

import LineChart from "@/modules/project/components/chart/LineChart.vue";
import BarLineMixinChart from "@/modules/project/components/chart/BarLineMixinChart.vue";

import { TaskType, AutonnLogTitle } from "@/shared/enums";

export default {
  components: { TrainContainer, ChartContainer, LineChart, BarLineMixinChart },

  props: {
    train: {
      default: null
    },

    trainLastSteps: {
      default: () => []
    },

    val: {
      default: null
    },

    valLastSteps: {
      default: () => []
    },

    epochSummary: {
      default: () => []
    }
  },

  data() {
    return {
      TaskType,

      trainContainerDefault: {
        summary: {
          first: { title: "", value: "" },
          second: { title: "", value: "" }
        },
        info: {
          left: { title: "", value: "" },
          center: { title: "", value: "" },
          right: { title: "", value: "" },
          result: { title: "", value: "" }
        },
        title: "",
        progressPercent: 0,
        progressTime: "",
        iter: 0
      },

      map50Datasets: [],
      mapDatasets: [],
      trainDatasets: [],
      totalTimeDatasets: [],
      epochTimeDatasets: []
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"]),

    taskType() {
      return this.project?.task_type || TaskType.DETECTION;
    },

    epochProgress() {
      if (this.train === null) return 0;
      else {
        return this.getPercent(this.train?.epoch || 0, this.train?.total_epoch || 0);
      }
    },

    trainLossLog() {
      if (this.train === null) {
        return this.trainContainerDefault;
      } else {
        return {
          summary: {
            first: { title: "EPOCH", value: `${this.train.epoch || 0}/${this.train.total_epoch || 0}` },
            second: { title: "GPU Mem", value: this.train.gpu_mem || "" }
          },
          info: {
            left: {
              title: AutonnLogTitle[this.taskType].train.left,
              value: this.fixedOrData(this.train.box || "")
            },
            center: {
              title: AutonnLogTitle[this.taskType].train.center,
              value: this.fixedOrData(this.train.obj || "")
            },
            right: {
              title: AutonnLogTitle[this.taskType].train.right,
              value: this.fixedOrData(this.train.cls || "")
            },
            result: {
              title: AutonnLogTitle[this.taskType].train.result,
              value: this.fixedOrData(this.train.total || "")
            }
          },
          title: "TRAINING LOSS",
          progressPercent: this.getPercent(this.train?.step || 0, this.train?.total_step || 0),
          progressTime: this.train.time,
          iter: 0
        };
      }
    },
    valAccuracyLog() {
      if (this.val === null) {
        return this.trainContainerDefault;
      } else {
        return {
          summary: {
            first: { title: "Images", value: Number(this.val.images).toLocaleString() || "" },
            second: { title: "Lables", value: Number(this.defaultOrData(this.val.labels, "")).toLocaleString() }
          },
          info: {
            left: {
              title: AutonnLogTitle[this.taskType].val.left,
              value: this.fixedOrData(this.defaultOrData(this.val.P, ""), 5)
            },
            center: {
              title: AutonnLogTitle[this.taskType].val.center,
              value: this.fixedOrData(this.defaultOrData(this.val.R, ""), 5)
            },
            right: {
              title: AutonnLogTitle[this.taskType].val.right,
              value: this.fixedOrData(this.defaultOrData(this.val.mAP50, ""), 7)
            },
            result: {
              title: AutonnLogTitle[this.taskType].val.result,
              value: this.fixedOrData(this.defaultOrData(this.val.mAP50_95, ""), 7)
            }
          },
          title: "VALIDATION ACCURACY",
          progressPercent: this.getPercent(this.val?.step || 0, this.val?.total_step || 0),
          progressTime: this.val.time,
          iter: 0
        };
      }
    },

    chartXAxisLabels() {
      if (this.epochSummary.length <= 0) return [];
      return [
        ...Array.from(
          { length: Math.min(this.epochSummary.length + 30, this.epochSummary[0].total_epoch) },
          () => 0
        ).map((q, index) => index + 1)
      ];
    },

    lossChartData() {
      if (!this.train) {
        return { labels: [], datasets: [] };
      }
      return {
        labels: this.chartXAxisLabels,
        datasets: [{ label: "Loss", backgroundColor: "#f87979", data: this.trainDatasets }]
      };
    },

    mAP50ChartData() {
      return {
        labels: this.chartXAxisLabels,
        datasets: [{ label: "mAP@0.5", backgroundColor: "#f87979", data: this.map50Datasets }]
      };
    },

    mAPChartData() {
      return {
        labels: this.chartXAxisLabels,
        datasets: [{ label: "mAP", backgroundColor: "#f87979", data: this.mapDatasets }]
      };
    },

    classificationChartData() {
      return {
        labels: this.chartXAxisLabels,
        datasets: [{ label: "Top-1 Accuracy", backgroundColor: "#f87979", data: this.mapDatasets }]
      };
    },

    elapsedTimeChartData() {
      return {
        labels: this.chartXAxisLabels,
        datasets: [
          { type: "line", label: "Total time", backgroundColor: "#000000", data: this.totalTimeDatasets },
          { type: "bar", label: "Epoch time", backgroundColor: "#f87979", data: this.epochTimeDatasets }
        ]
      };
    }
  },

  watch: {
    epochSummary() {
      const _map50Datasets = [];
      const _mapDatasets = [];
      const _trainDatasets = [];
      const _totalTimeDatasets = [];
      const _epochTimeDatasets = [];

      this.epochSummary.forEach(element => {
        if (element.train_loss_total === "Infinity") console.log("element", element);
        _map50Datasets.push(Number(element.val_acc_map50));
        _mapDatasets.push(Number(element.val_acc_map));
        _trainDatasets.push(Number(element.train_loss_total));
        _totalTimeDatasets.push(Number(element.total_time));
        _epochTimeDatasets.push(Number(element.epoch_time));
      });

      this.map50Datasets = _map50Datasets;
      this.mapDatasets = _mapDatasets;
      this.trainDatasets = _trainDatasets;
      this.totalTimeDatasets = _totalTimeDatasets;
      this.epochTimeDatasets = _epochTimeDatasets;
    }
  },

  methods: {
    defaultOrData(data, defaultText = "") {
      if (data === null) {
        return defaultText;
      } else {
        return data;
      }
    },

    fixedOrData(data, fixed = 5) {
      if (!isNaN(data)) {
        if (Number(data) === 0) {
          return 0;
        }

        return parseFloat(Number(data).toFixed(fixed)).toLocaleString("ko-KR", { maximumFractionDigits: fixed });
      } else {
        return data;
      }
    },

    getPercent(step, totalStep) {
      if (totalStep === 0) return 0;
      return this.fixedOrData((step / totalStep) * 100, 2);
    }
  }
};
</script>
<style lang="css" scoped>
.train-grid {
  display: grid;

  grid-template-rows: repeat(3, auto);
  grid-template-columns: 120px repeat(4, auto);

  grid-row-gap: 32px;
  grid-column-gap: 8px;
}
</style>
