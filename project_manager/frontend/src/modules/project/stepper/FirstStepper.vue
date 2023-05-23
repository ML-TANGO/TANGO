<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%">
    <div>
      <p class="text-h5 mb-10" style="color: #4a80ff">Project Informations</p>
      <v-text-field :value="project?.project_name" outlined dense label="Project Name" @change="onChangeName" />
      <v-text-field
        :value="project?.project_description"
        outlined
        dense
        hide-details
        label="Project Description"
        @change="onChangeDes"
      />
    </div>
    <div class="d-flex justify-end">
      <v-btn v-if="project?.id" class="ma-0 pa-0" text style="color: #4a80ff" @click="next"> NEXT </v-btn>
      <v-btn v-else class="ma-0 pa-0" text style="color: #4a80ff" @click="next"> CREATE </v-btn>
    </div>
  </div>
</template>
<script>
import { mapState } from "vuex";
import { ProjectNamespace } from "@/store/modules/project";

import { createProject, updateProjectName, updateProjectDescription } from "@/api";
export default {
  data() {
    return {
      projectName: "",
      projectDescription: ""
    };
  },

  computed: {
    ...mapState(ProjectNamespace, ["project"])
  },

  mounted() {
    this.projectName = this.project?.project_name;
    this.projectDescription = this.project.project_description;
  },

  methods: {
    async next() {
      try {
        if (this.projectName === "") {
          this.$swal("Project", "Project 이름을 입력해 주세요.", "error");
          return;
        } else if (this.projectDescription === "") {
          this.$swal("Project", "Project 설명을 입력해 주세요.", "error");
          return;
        }

        if (this.project?.id) {
          await updateProjectName(this.project.id, this.projectName);
          await updateProjectDescription(this.project.id, this.projectDescription);
          this.$emit("next", {
            ...this.project,
            project_name: this.projectName,
            project_description: this.projectDescription
          });
        } else {
          const res = await createProject(this.projectName, this.projectDescription);
          if (res.result === false) {
            this.$swal("Project 이름 중복", "이름 변경 후 다시 시도해 주세요.", "error");
            return;
          }
          console.log("project create ..... result : ", res);
          this.$emit("next", {
            id: res.id,
            project_name: this.projectName,
            project_description: this.projectDescription
          });
        }
        // this.$emit("next", { project_name: this.projectName, project_description: this.projectDescription });
      } catch {
        this.$swal("Project 생성 오류", "다시 시도해 주세요.", "error");
      }
    },

    onChangeDes(projectDescription) {
      this.projectDescription = projectDescription;
    },

    onChangeName(projectName) {
      this.projectName = projectName;
    }
  }
};
</script>
<style lang="scss" scoped></style>
