<template lang="">
  <div class="gird--view">
    <div class="left-view">
      <div style="grid-row: 1/2; position: relative; left: 25px" class="d-flex">
        <v-img :src="logo_3" :width="55" :max-width="55" contain></v-img>
        <div
          style="color: rgb(68, 143, 255); font-size: 3rem; font-weight: bold; margin-left: 8px"
          class="d-flex align-center mt-1"
        >
          TANGO
        </div>
      </div>
      <div style="width: 100%; margin-top: -10px" class="d-flex justify-center">
        <v-img :src="Tango_login" style="max-width: 80%" contain></v-img>
      </div>
    </div>
    <div class="right-view">
      <div class="login-form">
        <div class="login-header" style="text-align: center; font-weight: bold; color: rgb(48, 48, 48)">TANGO</div>

        <div class="d-flex flex-column border-none" style="gap: 5px">
          <v-text-field
            v-model="id"
            outlined
            :placeholder="'ID'"
            hide-details
            :background-color="'#d3d3d3'"
            @keydown.enter="onLogin"
          />
          <v-text-field
            v-model="pw"
            outlined
            :placeholder="'Password'"
            hide-details
            :background-color="'#d3d3d3'"
            type="password"
            @keydown.enter="onLogin"
          />
        </div>

        <div class="font-arial" style="margin-top: 30px">
          <v-btn
            style="width: 100%; height: 50px; font-size: 16px"
            class="text-capitalize"
            :color="'#4a80ff'"
            dark
            @click="onLogin"
          >
            Log in
          </v-btn>
        </div>
        <div style="margin-top: 10px; font-size: 14px">
          <span>Don't have account?</span>
          <span style="margin-left: 5px">
            <span class="sign_up_link" style="color: #4a80ff; cursor: pointer" @click="onNavigateCA">Register now</span>
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import Cookies from "universal-cookie"; // MIT

import Tango_login from "@/assets/Tango_login.png";
import logo_3 from "@/assets/icon_3x/Tango_logo.png";

import { userLoginAPI } from "@/api";
export default {
  data() {
    return {
      Tango_login,
      logo_3,
      id: "",
      pw: ""
    };
  },

  methods: {
    /** 로그인 버튼 클릭 이벤트 */
    async onLogin() {
      const userInfo = await userLoginAPI(this.id, this.pw);
      console.log("userInfo", userInfo);

      if (userInfo.result) {
        const content = JSON.parse(userInfo.content);

        const cookies = new Cookies();
        cookies.set("TANGO_TOKEN", content["access_token"], { path: "/" });
        cookies.set("userinfo", this.id, { path: "/" });

        this.$router.go();
      } else {
        this.$swal("로그인 실패", "ID, PW를 확인해주세요.", "error");
      }
    },

    /** 회원가입 페이지로 이동 이벤트 */
    onNavigateCA() {
      this.$router.push("create-account");
    }
  }
};
</script>
<style lang="scss" scoped>
.gird--view {
  display: grid;
  grid-template-columns: 1fr 1fr;
  background-color: #fff;

  height: 100%;
}

.left-view {
  display: grid;
  grid-template-rows: 10fr 80fr 10fr;
  background: #fff;
}

.right-view {
  display: grid;
  grid-template-columns: 20fr 50fr 20fr;
  grid-template-rows: 30fr 60fr 30fr;
  background: #303030;
}

.login-form {
  grid-column: 2/3;
  grid-row: 2/2;
  background: #fff;
  color: #000;
  height: 560px auto;
  width: 540px;
  border-radius: 7px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
  padding: 70px;
}

.login-header {
  margin-bottom: 28px;
  font-size: 40px;
  font-weight: 500;
}
</style>

<style>
.border-none
  .v-text-field--outlined:not(.v-input--is-focused):not(.v-input--has-state)
  > .v-input__control
  > .v-input__slot
  fieldset {
  border: none;
}

.font-arial {
  font-family: Arial, Helvetica, sans-serif;
}
</style>
