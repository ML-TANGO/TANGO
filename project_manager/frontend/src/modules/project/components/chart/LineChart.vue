<template>
  <LineChartGenerator
    :options="chartOptions"
    :data="chartData"
    :chart-id="chartId"
    :dataset-id-key="datasetIdKey"
    :plugins="plugins"
    :css-classes="cssClasses"
    :styles="styles"
  />
  <!-- :width="width"
    :height="height" -->
</template>

<script>
import { Line as LineChartGenerator } from "vue-chartjs";

import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  LinearScale,
  CategoryScale,
  PointElement
} from "chart.js";

ChartJS.register(Title, Tooltip, Legend, LineElement, LinearScale, CategoryScale, PointElement);

export default {
  name: "LineChart",
  components: {
    LineChartGenerator
  },
  props: {
    chartId: {
      type: String,
      default: "line-chart"
    },
    datasetIdKey: {
      type: String,
      default: "label"
    },
    width: {
      type: Number,
      default: 300
    },
    height: {
      type: Number,
      default: 300
    },
    cssClasses: {
      default: "",
      type: String
    },
    styles: {
      type: Object,
      default: () => {}
    },
    plugins: {
      type: Array,
      default: () => []
    },
    xTitle: {
      type: String,
      default: ""
    },
    yTitle: {
      type: String,
      default: ""
    },
    chartData: {
      type: Object,
      default: () => ({
        labels: [1, 2, 3, 4],
        datasets: [
          {
            label: "Loss",
            backgroundColor: "#f87979",
            data: [0.875, 0.63, 0.9123, 0.4845]
          }
        ]
      })
    }
  },
  data() {
    return {};
  },

  computed: {
    chartOptions() {
      return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                let label = context.dataset.label || "";

                if (label) {
                  label += ": ";
                }
                if (context.parsed.y !== null) {
                  label += parseFloat(context.parsed.y.toFixed(8));
                }
                return label;
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: this.xTitle,
              align: "end"
            }
          },
          y: {
            // max: 1,
            min: 0,
            title: {
              display: true,
              text: this.yTitle,
              align: "end"
            }
          }
        }
      };
    }
  }
};
</script>
