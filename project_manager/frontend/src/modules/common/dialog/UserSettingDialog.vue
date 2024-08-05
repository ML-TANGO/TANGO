<template>
  <v-dialog v-model="isOpen" width="800px" persistent>
    <v-card width="800px" height="400px">
      <v-btn icon absolute right top @click="onClose"><v-icon color="#ccc" size="34">mdi-close-circle</v-icon> </v-btn>

      <v-card-title>Setting</v-card-title>
      <v-divider></v-divider>
      <div style="height: calc(100% - 60px)" class="d-flex">
        <v-navigation-drawer left>
          <v-list dense>
            <v-list-item
              v-for="(item, index) in items"
              :key="item.title"
              @click="onChangeTab(index)"
              style="transition: 0.3s"
              :class="currentTab === index ? 'active' : ''"
            >
              <v-list-item-icon>
                <v-icon :style="{ color: currentTab === index ? 'white' : '' }">{{ item.icon }}</v-icon>
              </v-list-item-icon>

              <v-list-item-content>
                <v-list-item-title>{{ item.title }}</v-list-item-title>
              </v-list-item-content>
            </v-list-item>
          </v-list>
        </v-navigation-drawer>
        <div style="position: relative; overflow-y: auto; width: 100%" class="pa-2">
          <div v-if="currentTab === 0">
            <div class="d-flex align-center" style="gap: 8px">
              <div>Progress 갱신 주기 :</div>
              <div style="width: 60px">
                <v-text-field
                  :value="progressTime"
                  @change="onChangeProgress"
                  dense
                  outlined
                  hide-details
                  type="number"
                  max="20"
                  :key="`progressInput-${progressTimeKey}`"
                ></v-text-field>
              </div>
              초
            </div>

            <div class="mt-3 d-flex align-center" style="gap: 8px">
              <div>Auto NN 시각화 갱신 주기 :</div>
              <div style="width: 60px">
                <v-text-field
                  :value="autonnTime"
                  @change="onChangeAutonn"
                  dense
                  outlined
                  hide-details
                  type="number"
                  max="20"
                  :key="`autonnInput-${autonnTimeKey}`"
                ></v-text-field>
              </div>
              초
            </div>
          </div>
        </div>
      </div>
    </v-card>
  </v-dialog>
</template>
<script>
import { getUserIntervalTime, setUserIntervalTime } from "@/api";

export default {
  data() {
    return {
      isOpen: false,
      items: [{ title: "Project Setting", icon: "mdi-cog" }],
      currentTab: 0,

      progressTime: 10,
      progressTimeKey: 0,

      autonnTime: 1,
      autonnTimeKey: 0
    };
  },

  mounted() {
    this.getUserIntervalSetting();
  },

  methods: {
    async getUserIntervalSetting() {
      const intervalTimes = await getUserIntervalTime();

      this.progressTime = intervalTimes["project_status"];
      this.autonnTime = intervalTimes["autonn_status"];

      console.log("this.progressTime", this.progressTime);
      console.log("this.autonnTime", this.autonnTime);
    },

    onSave() {},

    onChangeTab(index) {
      this.currentTab = index;
    },

    onClose() {
      this.isOpen = false;
    },

    onChangeProgress(time) {
      this.progressTime = time > 20 ? 20 : time <= 0 ? 1 : time;
      this.progressTimeKey++;

      this.$EventBus.$emit("updateProgressTime", this.progressTime);

      this.update();
    },

    onChangeAutonn(time) {
      this.autonnTime = time > 20 ? 20 : time <= 0 ? 1 : time;
      this.autonnTimeKey++;

      this.$EventBus.$emit("updateAutonnTime", this.autonnTime);

      this.update();
    },

    update() {
      setUserIntervalTime(Number(this.progressTime), Number(this.autonnTime));
    }
  }
};
</script>
<style lang="css" scoped>
.active {
  background-color: #2a98fb;
  color: white !important;
}
</style>
