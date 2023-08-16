<template>
  <div id="side-bar" class="side-transition">
    <v-card :width="mini ? 90 : 250" class="rounded-0 side-transition pl-4" height="100%" style="background: #303030">
      <!-- <v-btn icon class="mt-10 mb-10" style="margin-left: 8px">
        <v-img :src="Tango_logo" width="15" max-width="30" contain></v-img>
      </v-btn> -->

      <v-list dense style="background: #303030" class="pr-7" :elevation="0" nav v-if="mini">
        <v-list-item @click="onNavigate($event, '/')">
          <v-list-item-icon>
            <v-img :src="Tango_logo" width="15" max-width="30" contain></v-img>
          </v-list-item-icon>
          <v-list-item-content style="margin-left: -13px">
            <div style="font-size: 24px; color: #4a80ff" class="font-weight-bold">TANGO</div>
          </v-list-item-content>
        </v-list-item>
      </v-list>

      <v-list dense style="background: #303030" class="pr-7 mt-5" :elevation="0" nav v-else>
        <v-list-item @click="onNavigate($event, items[0].address)">
          <v-list-item-content style="gap: 15px" class="d-flex flex-column align-center">
            <v-img :src="Tango_logo" width="60" max-width="60" contain></v-img>
            <div style="font-size: 38px; color: #4a80ff" class="font-weight-bold">TANGO</div>
          </v-list-item-content>
        </v-list-item>
      </v-list>

      <v-list dense style="background: #303030" class="pr-7" :elevation="0" nav>
        <div class="d-flex flex-column non" style="gap: 15px">
          <div v-for="item in items" :key="item.title" class="non">
            <v-tooltip right content-class="tooltip-right tango" nudge-right="45" :disabled="!mini">
              <template v-slot:activator="{ on, attrs }">
                <v-hover v-slot="{ hover }">
                  <v-list-item
                    v-bind="attrs"
                    v-on="on"
                    class="non side-transition"
                    @click.stop="onNavigate($event, item.address)"
                    :style="{ backgroundColor: hover || currentPath.includes(item.image) ? '#515559' : '#303030' }"
                  >
                    <v-list-item-icon>
                      <v-img :src="routingImage(item.image)" width="15" max-width="20" contain></v-img>
                    </v-list-item-icon>

                    <v-list-item-content style="margin-left: -15px">
                      <v-list-item-title :style="{ color: currentPath.includes(item.image) ? '#d5d8db' : '#6c7890' }">
                        {{ item.title }}
                      </v-list-item-title>
                    </v-list-item-content>
                  </v-list-item>
                </v-hover>
              </template>
              <span style="font-size: 12px">{{ item.title }}</span>
            </v-tooltip>
          </div>
        </div>
      </v-list>
    </v-card>
    <!-- :style="{ left: !mini ? '235px' : '100px' }" -->
    <div class="mini--btn mt-10 side-transition" :style="{ left: !mini ? '235px' : '75px' }">
      <v-btn v-if="mini" icon style="background: #3f3f3f" width="30" height="30" @click="mini = false">
        <v-icon color="white">mdi-chevron-right</v-icon>
      </v-btn>
      <v-btn v-else icon style="background: #3f3f3f" width="30" height="30" @click="mini = true">
        <v-icon color="white">mdi-chevron-left</v-icon>
      </v-btn>
    </div>
  </div>
</template>
<script>
import project_mgmt from "@/assets/icon_3x/project_mgmt.png";
import dataMgmt from "@/assets/icon_3x/data_mgmt@3x.png";
import target from "@/assets/icon_3x/target@3x.png";
import visualization from "@/assets/icon_3x/visualization.png";

import project_mgmt_on from "@/assets/icon_3x/project_mgmt_on@3x.png";
import dataMgmt_on from "@/assets/icon_3x/data_mgmt_on@3x.png";
import target_on from "@/assets/icon_3x/target_on@3x.png";
import visualization_on from "@/assets/icon_3x/visualization_on.png";

import Tango_logo from "@/assets/icon_3x/Tango_logo.png";

import "@/sass/tooltip.scss";

export default {
  data() {
    return {
      mini: false,
      items: [
        { title: "Project Management", image: "project", address: "/project" },
        { title: "Target Management", image: "target", address: "/target" },
        { title: "Data Management", image: "data", address: "/data" },
        { title: "Visualization", image: "visualization", address: "/visualization" }
      ],
      routerImages: {
        project: project_mgmt,
        project_on: project_mgmt_on,
        target: target,
        target_on: target_on,
        data: dataMgmt,
        data_on: dataMgmt_on,
        visualization: visualization,
        visualization_on: visualization_on
      },
      drawer: true,
      Tango_logo
    };
  },

  computed: {
    currentPath() {
      return this.$route.path;
    }
  },

  watch: {
    mini() {
      console.log("mini", this.mini);
      this.$emit("mini", this.mini);
    }
  },

  methods: {
    onNavigate(e, address) {
      e.preventDefault();
      e.cancelBubble = true;

      if (!this.currentPath.includes(address)) {
        this.$router.push(address);
      }
    },

    routingImage(image) {
      const img = this.currentPath.includes(image) ? image + "_on" : image;
      return this.routerImages[img];
    }
  }
};
</script>
<style lang="scss" scoped>
#side-bar {
  z-index: 185;
  height: 100%;
  display: flex;

  position: relative;
}

.side-transition {
  transition-duration: 0.2s !important;
  transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1) !important;
  transition-property: transform, visibility, width, all !important;
}
</style>

<style>
.non .v-list-item--link:before {
  content: none;
}

.mini--btn {
  z-index: 190;
  position: fixed;
}
</style>
