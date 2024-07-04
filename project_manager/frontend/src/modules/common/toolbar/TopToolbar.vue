<template lang="">
  <div style="width: 100%; height: 60px; min-height: 60px" class="top-layout">
    <h3
      style="grid-column: 1; margin-left: 43px; color: #0000008a; letter-spacing: 3px"
      v-if="$route.name !== 'ProjectDetail'"
    >
      {{ $route.name.toUpperCase() }}
    </h3>
    <h3 style="grid-column: 1; margin-left: 43px; color: #0000008a; letter-spacing: 3px" v-else>
      {{ project.project_name }}
    </h3>
    <div style="grid-column: 2">
      <!-- <v-btn dark icon>
        <v-icon color="black">mdi-bell-outline</v-icon>
      </v-btn> -->
    </div>
    <div style="grid-column: 3">
      <v-menu bottom offset-y :rounded="'lg'" :nudge-bottom="15" :nudge-left="30">
        <template v-slot:activator="{ on, attrs }">
          <div v-bind="attrs" v-on="on">
            <!-- <div style="font-size: 16px; color: black; text-align: start">
              <div class="font-weight-bold;" style="font-size: 13px">gmlee</div>
              <div style="font-size: 11px; color: #ccc">gmlee@teslasystem.co.kr</div>
            </div> -->
            <v-btn icon>
              <v-icon>mdi-menu</v-icon>
            </v-btn>
          </div>
        </template>

        <v-list dense style="width: 200px" class="px-3">
          <!-- <v-list-item>
            <v-list-item-icon>
              <v-icon color="black">mdi-cog</v-icon>
            </v-list-item-icon>
            <v-list-item-content style="margin-left: -15px">
              <v-list-item-title> setting </v-list-item-title>
            </v-list-item-content>
          </v-list-item>
          <v-divider class="my-3" /> -->
          <v-list-item @click="logout">
            <v-list-item-icon>
              <v-icon color="black">mdi-logout</v-icon>
            </v-list-item-icon>
            <v-list-item-content style="margin-left: -15px">
              <v-list-item-title> Logout </v-list-item-title>
            </v-list-item-content>
          </v-list-item>
        </v-list>
      </v-menu>
    </div>
  </div>
</template>
<script>
import { mapState } from "vuex";
import { ProjectNamespace } from "@/store/modules/project";

import Cookies from "universal-cookie";
export default {
  computed: {
    ...mapState(ProjectNamespace, ["project"])
  },
  methods: {
    logout() {
      const cookies = new Cookies();
      cookies.remove("userinfo", { path: "/" });
      cookies.remove("TANGO_TOKEN", { path: "/" });
      this.$router.go();
    }
  }
};
</script>
<style lang="scss" scoped>
.top-layout {
  display: grid;
  grid-template-columns: auto 60px 60px;
  align-items: center;
}
</style>
