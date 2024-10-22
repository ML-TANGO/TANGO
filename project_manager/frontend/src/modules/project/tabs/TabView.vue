<template>
  <v-card style="width: 100%; min-height: calc(100vh - 106px)" class="pa-5">
    <div class="d-flex ml-9" style="gap: 35px; flex-wrap: wrap">
      <TabCard v-for="(card, cindex) in items" :key="'card' + cindex" :projectInfo="card" :status="status(card)" />
    </div>
  </v-card>
</template>
<script>
import TabCard from "../TabCard.vue";
import { ProjectStatus } from "@/shared/enums";
export default {
  components: { TabCard },
  props: {
    items: {
      default: () => []
    }
  },
  methods: {
    status(info) {
      // if (!info?.dataset || info?.dataset === "") return { title: "Dataset", color: "#FF3D54" };
      if (info.container !== "" && info.container !== "init") {
        return { title: info?.container_status.toUpperCase(), color: "#4a80ff" };
      } else if (info?.container_status === "fail") return { title: info?.container_status, color: "#FF3D54" };
      else if (info?.container_status === ProjectStatus.COMPLETED) return { title: info?.container, color: "#4a80ff" };
      else if (info?.container_status === "" && info?.container === "") return { title: "READY", color: "#4a80ff" };
      else if (!info?.target_id || info?.target_id === "") return { title: "Target", color: "#FF3D54" };
      else if (!info?.task_type || info?.task_type === "") return { title: "Task Type", color: "#FF3D54" };
      else if (!info?.nas_type || info?.nas_type === "") return { title: "Nas Type", color: "#FF3D54" };
      else if (!info?.deploy_weight_level || info?.deploy_weight_level === "")
        return { title: "Weight Level", color: "#FF3D54" };
      else if (!info?.deploy_precision_level || info?.deploy_precision_level === "")
        return { title: "Precision Level", color: "#FF3D54" };
      else if (!info?.deploy_user_edit || info?.deploy_user_edit === "")
        return { title: "User Edit", color: "#FF3D54" };
      else if (!info?.deploy_output_method || info?.deploy_output_method === "")
        return { title: "Output Method", color: "#FF3D54" };
      else return { title: "READY", color: "#4a80ff" };
    }
  }
};
</script>
<style></style>
