<template lang="">
  <v-card style="height: 100% overflow: none">
    <v-tabs v-model="tab" align-with-title>
      <v-tabs-slider color="#4a80ff"></v-tabs-slider>
      <!-- style="width: 160px" -->
      <v-tab v-for="item in items" :key="item">
        {{ item }}
      </v-tab>
    </v-tabs>
    <v-divider />
    <v-tabs-items v-model="tab" style="height: calc(100vh - 106px); overflow: overlay">
      <!-- All Status -->
      <v-tab-item> <AllStatusTab :itemByTabs="this.projectsByTab" /> </v-tab-item>
      <!-- Ready -->
      <v-tab-item>
        <TabView v-if="isData(this.projectsByTab['Preparing'])" :items="this.projectsByTab['Preparing']" />
        <NoResultTab v-else />
      </v-tab-item>
      <!-- BMS -->
      <v-tab-item>
        <TabView v-if="isData(this.projectsByTab['BMS'])" :items="this.projectsByTab['BMS']" />
        <NoResultTab v-else />
      </v-tab-item>
      <!-- Visualization -->
      <v-tab-item>
        <TabView v-if="isData(this.projectsByTab['Visualization'])" :items="this.projectsByTab['Visualization']" />
        <NoResultTab v-else />
      </v-tab-item>
      <!-- AUTO NN -->
      <v-tab-item>
        <TabView v-if="isData(this.projectsByTab['Auto NN'])" :items="this.projectsByTab['Auto NN']" />
        <NoResultTab v-else />
      </v-tab-item>
      <!-- IMAGE GEN -->
      <v-tab-item>
        <TabView v-if="isData(this.projectsByTab['Code Gen'])" :items="this.projectsByTab['Code Gen']" />
        <NoResultTab v-else />
      </v-tab-item>
      <!-- IMAGE deploy -->
      <v-tab-item>
        <TabView v-if="isData(this.projectsByTab['Image Deploy'])" :items="this.projectsByTab['Image Deploy']" />
        <NoResultTab v-else />
      </v-tab-item>
    </v-tabs-items>

    <ProjectCreateDialog :step="step" @stepChange="onStepChange" @close="close">
      <template v-slot:btn>
        <v-btn color="tango" dark absolute style="top: 4px; right: 30px" height="40" width="180">
          NEW PROJECT&nbsp;<v-icon :size="20">mdi-plus</v-icon>
        </v-btn>
      </template>
    </ProjectCreateDialog>
  </v-card>
</template>
<script>
import { mapMutations } from "vuex";
import { ProjectNamespace, ProjectMutations } from "@/store/modules/project";

import AllStatusTab from "@/modules/project/tabs/AllStatusTab.vue";
import TabView from "@/modules/project/tabs/TabView.vue";
import NoResultTab from "@/modules/project/tabs/NoResultTab.vue";
import ProjectCreateDialog from "@/modules/project/ProjectCreateDialog.vue";

import { getProjectList } from "@/api";

export default {
  components: {
    AllStatusTab,
    TabView,
    ProjectCreateDialog,
    NoResultTab
  },
  data() {
    return {
      tab: null,
      step: 1,
      items: ["All Status", "Preparing", "BMS", "Visualization", "Auto NN", "Code Gen", "Image Deploy"],
      defaultValue: { Preparing: [], BMS: [], Visualization: [], "Auto NN": [], "Code Gen": [], "Image Deploy": [] },
      projectsByTab: {},
      tabItems: [
        { key: "Preparing", allowed: ["", "init"] },
        { key: "BMS", allowed: ["bms"] },
        { key: "Visualization", allowed: ["visualization", "viz2code"] },
        { key: "Auto NN", allowed: ["autonk", "yoloe", "autobb", "autonn-resnet"] },
        { key: "Code Gen", allowed: ["codeGen"] },
        { key: "Image Deploy", allowed: ["imagedeploy"] }
        // { key: "Run Image", allowed: ["run_image"] }
      ]
    };
  },

  async created() {
    await this.initProjectList();

    this.$EventBus.$on("deleteProject", async () => {
      await this.initProjectList();
    });

    this.$EventBus.$on("projectDialogclose", async () => {
      await this.close();
    });
  },

  mounted() {
    this.INIT_PROJECT();
  },

  methods: {
    ...mapMutations(ProjectNamespace, {
      INIT_PROJECT: ProjectMutations.INIT_PROJECT
    }),

    async initProjectList() {
      const projectList = await getProjectList();

      this.projectsByTab = {
        ...this.defaultValue,
        ...projectList.reduce((acc, val) => {
          const container = this.tabItems.find(q => q.allowed.includes(val.container)).key;
          if (!Object.keys(acc).includes(container)) {
            acc[container] = [];
          }
          acc[container].push(val);
          return acc;
        }, {})
      };
    },
    onStepChange(step) {
      this.step = step;
    },

    async close() {
      await this.initProjectList();
    },

    isData(data) {
      return data?.length && data?.length > 0;
    }
  }
};
</script>
<style lang="scss"></style>
