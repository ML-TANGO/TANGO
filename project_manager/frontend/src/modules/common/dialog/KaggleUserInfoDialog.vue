<template>
  <v-dialog v-model="isOpen" width="35%" persistent>
    <v-card width="100%">
      <v-card-title>Kaggle API</v-card-title>
      <v-card-subtitle class="pt-2">Kaggle 데이터셋을 다운로드 하려면 정보를 입력해주세요.</v-card-subtitle>
      <v-card-text>
        <v-container fluid>
          <v-row>
            <v-col cols="3" class="d-flex align-center pa-0">
              <v-subheader>username</v-subheader>
            </v-col>
            <v-col cols="9" class="pa-0">
              <v-text-field placeholder="username" v-model="username"></v-text-field>
            </v-col>
          </v-row>

          <v-row>
            <v-col cols="3" class="d-flex align-center pa-0">
              <v-subheader>key</v-subheader>
            </v-col>
            <v-col cols="9" class="pa-0">
              <v-text-field placeholder="key" v-model="key"></v-text-field>
            </v-col>
          </v-row>
        </v-container>
      </v-card-text>
      <v-card-actions class="px-5">
        <v-spacer></v-spacer>
        <v-btn color="error" dark @click="onClose">취소</v-btn>
        <v-btn color="tango" dark @click="onSave">저장</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>
<script>
import { checkExistKaggleInfo, createUserKaggleInfo } from "@/api";
export default {
  data() {
    return {
      isOpen: false,
      username: "",
      key: ""
    };
  },

  mounted() {
    this.checkExistKaggleInfoHandler();
  },

  methods: {
    async checkExistKaggleInfoHandler() {
      const res = await checkExistKaggleInfo();
      if (res?.isExist) {
        this.username = res.username;
        this.key = res.key;
      }
    },

    onSave() {
      createUserKaggleInfo(this.username, this.key).then(() => {
        this.$emit("restart");

        this.isOpen = false;
      });
    },

    onClose() {
      this.isOpen = false;
      this.$router.push("/");
    }
  }
};
</script>
<style></style>
