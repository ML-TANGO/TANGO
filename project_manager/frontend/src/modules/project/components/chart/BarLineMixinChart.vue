<template>
  <Bar
    :options="chartOptions"
    :data="chartData"
    :chart-id="chartId"
    :dataset-id-key="datasetIdKey"
    :plugins="plugins"
    :css-classes="cssClasses"
    :styles="styles"
  />
</template>
<script>
import {
  Chart as ChartJS,
  Title,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
  BarElement,
  CategoryScale,
  Filler,
  LinearScale
} from "chart.js";
import { Bar } from "vue-chartjs";

ChartJS.register(CategoryScale, PointElement, LineElement, LinearScale, BarElement, Title, Tooltip, Filler, Legend);

export default {
  name: "LineChart",
  components: {
    Bar
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
            type: "line",
            label: "Loss12",
            backgroundColor: "#000000",
            data: [0.675, 0.863, 0.9123, 0.5845]
          },

          {
            type: "bar",
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
                  label += context.parsed.y.toFixed(2);
                  label += " s";
                  // label += context.parsed.y.toString() + " s";
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

<style></style>
