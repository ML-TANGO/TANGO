<template>
  <v-card style="width: 100%; min-height: calc(100vh - 106px)" class="pa-5">
    <TabBase
      v-for="(item, index) in Object.keys(itemByTabs)"
      :key="index"
      :count="itemByTabs[item].length"
      :title="item"
      :defaultBanner="true"
    >
      <template #content>
        <div class="d-flex" style="gap: 25px; flex-wrap: wrap">
          <TabCard
            v-for="card in itemByTabs[item]"
            :key="'card' + card.id"
            :projectInfo="card"
            :status="statusRes(card)"
          />
        </div>
      </template>
    </TabBase>
  </v-card>
</template>
<script>
import { mapMutations } from "vuex";
import TabBase from "../TabBase.vue";
import TabCard from "../TabCard.vue";
import { ProjectStatus, TaskType, LearningType } from "@/shared/enums";
export default {
  components: { TabBase, TabCard },

  props: {
    itemByTabs: {
      default: () => ({})
    }
  },

  data() {
    return {
      count: 0,
      totalCount: 0
    };
  },

  computed: {
    statusRes() {
      return info => {
        // Auto NN CL (Segmentation + Continual Learning) 프로젝트는
        // 자체 데이터셋을 사용하므로 데이터셋 체크를 스킵
        const isAutoNNCL =
          info?.task_type === TaskType.SEGMENTATION &&
          info?.learning_type === LearningType.CONTINUAL_LEARNING;
        
        // Chat과 Auto NN CL 프로젝트가 아닌 경우에만 데이터셋 체크
        if (!isAutoNNCL && info?.task_type !== TaskType.CHAT &&
          (!info?.dataset || info?.dataset === ""))
          return { title: "Dataset", color: "#FF3D54" };
        // Auto NN CL 프로젝트는 Target이 없어도 됨
        else if (!isAutoNNCL && (!info?.target_id || !info?.target_info))
          return { title: "Target", color: "#FF3D54" };
        else if (info.container !== "" && info.container !== "init")
          return { title: info?.container_status?.toUpperCase() || "READY", color: "#4a80ff" };
        else if (info?.container_status === ProjectStatus.FAILED)
          return { title: info?.container_status, color: "#FF3D54" };
        else if (info?.container_status === ProjectStatus.COMPLETED)
          return { title: info?.container, color: "#4a80ff" };
        else if (info?.container_status === "" && info?.container === "")
          return { title: "READY", color: "#4a80ff" };
        else if (!isAutoNNCL && (!info?.target_id || info?.target_id === ""))
          return { title: "Target", color: "#FF3D54" };
        else if (!info?.task_type || info?.task_type === "")
          return { title: "Task Type", color: "#FF3D54" };
        // Auto NN CL 프로젝트는 nas_type이 없어도 됨
        else if (!isAutoNNCL && (!info?.nas_type || info?.nas_type === ""))
          return { title: "Nas Type", color: "#FF3D54" };
        else if (!info?.deploy_weight_level || info?.deploy_weight_level === "")
          return { title: "Weight Level", color: "#FF3D54" };
        else if (!info?.deploy_precision_level || info?.deploy_precision_level === "")
          return { title: "Precision Level", color: "#FF3D54" };
        else if (!info?.deploy_user_edit || info?.deploy_user_edit === "")
          return { title: "User Edit", color: "#FF3D54" };
        else if (!info?.deploy_output_method || info?.deploy_output_method === "")
          return { title: "Output Method", color: "#FF3D54" };
        else return { title: "READY", color: "#4a80ff" };
      };
    }
  },

  watch: {
    itemByTabs() {
      this.totalCount = Object.keys(this.itemByTabs).reduce((acc, val) => {
        acc += this.itemByTabs[val].length;
        return acc;
      }, 0);
    }
  },

  methods: {
    ...mapMutations(["setLoding"])
  }
};
</script>
<style lang=""></style>
