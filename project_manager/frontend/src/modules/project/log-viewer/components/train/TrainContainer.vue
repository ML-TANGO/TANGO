<template>
  <div class="d-flex align-end" style="gap: 8px; width: 100%">
    <div>
      <div class="rect" :style="{ backgroundColor: primaryColor, color: 'white' }">{{ data.summary.first.title }}</div>
      <div class="rect mt-1" :style="{ border: `3px solid ${primaryColor}` }">{{ data.summary.first.value }}</div>
    </div>

    <div>
      <div class="rect" :style="{ backgroundColor: primaryColor, color: 'white' }">{{ data.summary.second.title }}</div>
      <div class="rect mt-1" :style="{ border: `3px solid ${primaryColor}` }">{{ data.summary.second.value }}</div>
    </div>

    <div style="width: 60%">
      <div class="d-flex align-center" style="margin-top: -20px">
        <div style="height: 2px; width: 100%" class="ml-3" :style="{ backgroundColor: primaryColor }"></div>
        <div style="font-size: 14px" class="mx-2" :style="{ color: primaryColor }">{{ data.title }}</div>
        <div style="height: 2px; width: 100%" class="mr-3" :style="{ backgroundColor: primaryColor }"></div>
      </div>

      <div
        class="rounded-rect"
        style="width: 100%; justify-content: space-between"
        :style="{ backgroundColor: primaryColor }"
      >
        <div class="d-flex" style="width: 100%">
          <div class="rounded-rect inner top first" :style="{ backgroundColor: secondaryColor }">
            {{ data.info.left.title }}
          </div>
          <div class="rounded-rect inner top" :style="{ backgroundColor: secondaryColor }">
            {{ data.info.center.title }}
          </div>
          <div class="rounded-rect inner top" :style="{ backgroundColor: secondaryColor }">
            {{ data.info.right.title }}
          </div>
        </div>

        <div style="text-align: right; width: 80px; color: white" class="mr-3">{{ data.info.result.title }}</div>
      </div>

      <div
        class="rounded-rect mt-1"
        style="width: 100%; justify-content: space-between"
        :style="{ border: `2px solid ${primaryColor}` }"
      >
        <div class="d-flex" style="width: 100%">
          <div class="rounded-rect inner bottom first" :style="{ border: `2px solid ${primaryColor}` }">
            {{ data.info.left.value }}
          </div>
          <div class="rounded-rect inner bottom" :style="{ border: `2px solid ${primaryColor}` }">
            {{ data.info.center.value }}
          </div>
          <div class="rounded-rect inner bottom" :style="{ border: `2px solid ${primaryColor}` }">
            {{ data.info.right.value }}
          </div>
        </div>

        <div style="text-align: right; width: 80px" class="mr-3">{{ data.info.result.value }}</div>
      </div>
    </div>

    <div style="width: 40%" class="d-flex align-end px-3">
      <div class="d-flex align-center" style="width: 100%; gap: 20px">
        <v-progress-linear :value="data.progressPercent" height="30" rounded>
          {{ data.progressPercent }}%
        </v-progress-linear>
        <div style="width: 230px; text-align: right">{{ data.progressTime }} / iter</div>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  props: {
    primaryColor: {
      default: "#ffffff"
    },

    secondaryColor: {
      default: "#ffffff"
    },

    data: {
      default: () => ({
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
      })
    }
  }
};
</script>
<style>
.rect {
  border-radius: 8px;
  height: 30px;
  width: 110px;

  font-size: 13px;

  display: flex;
  align-items: center;
  justify-content: center;
}

.rounded-rect {
  border-radius: 20px;
  height: 30px;

  display: flex;
  align-items: center;
  justify-content: center;
}

.inner {
  width: 27%;
  font-size: 13px;
}

.inner.top {
  color: white;
  border: 2px solid white;

  margin-left: -1px;
}

.inner.top.first {
  border: none;
  border: 2px solid white;
}

.inner.bottom {
  color: black;
}

.inner.bottom.first {
  margin-left: -2px;
}
</style>
