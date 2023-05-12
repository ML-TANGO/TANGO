<template>
  <v-app id="app">
    <v-main style="max-height: 100%">
      <div style="width: 100%; height: 100%" class="d-flex">
        <div v-if="!noSidebar.includes(currentLocation)"></div>
        <div class="d-flex flex-column" style="width: 100%; height: 100%; position: relative">
          <router-view />
        </div>
      </div>
    </v-main>
  </v-app>
</template>

<script>
import { mapState } from "vuex";

export default {
  name: "App",
  data() {
    return {
      noSidebar: ["/login", "/create-account", "/find-password", "/change-password"],
      mini: true
    };
  },

  computed: {
    ...mapState(["loading"]),
    currentLocation() {
      return this.$route.path;
    }
  },

  created() {
    const html = document.getElementsByTagName("html");
    html[0].style.overflowY = "hidden";
  },

  methods: {
    setMini(mini) {
      this.mini = mini;
    }
  }
};
</script>

<style>
body {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

*::-webkit-scrollbar {
  width: 8px; /* 스크롤바의 너비 */
  height: 8px; /* 스크롤바의 너비 */
}

*::-webkit-scrollbar-thumb {
  height: 30%; /* 스크롤바의 길이 */
  background: #c6c6c6; /* 스크롤바의 색상 */

  border-radius: 10px;
}

*::-webkit-scrollbar-track {
  background-color: transparent; /* 배경 색상 */
}
</style>
