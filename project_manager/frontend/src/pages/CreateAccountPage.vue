<template lang="">
  <div class="gird--view">
    <div class="left-view">
      <div style="grid-row: 1/2; position: relative; left: 25px" class="d-flex">
        <v-img :src="logo_3" :width="55" :max-width="55" contain></v-img>
        <div
          style="color: rgb(68, 143, 255); font-size: 3rem; font-weight: bold; margin-left: 8px"
          class="d-flex align-center"
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
        <div class="login-header" style="text-align: center; font-weight: bold; color: rgb(48, 48, 48)">
          Create Account
        </div>

        <div class="d-flex flex-column border-none" style="gap: 5px">
          <v-form ref="form" lazy-validation class="d-flex flex-column" style="height: 340px; gap: 5px">
            <div class="d-flex" style="gap: 15px">
              <v-text-field
                v-model="id"
                outlined
                :placeholder="'ID'"
                :background-color="'#d3d3d3'"
                :rules="userIDRule()"
              />
              <v-btn dark :color="'#4a80ff'" style="height: 56px" @click="onDuplicateIDCheck">Check</v-btn>
            </div>
            <v-text-field
              v-model="email"
              outlined
              :placeholder="'E-Mail'"
              :background-color="'#d3d3d3'"
              :rules="emailRule()"
            />
            <v-text-field
              v-model="pw"
              outlined
              :placeholder="'Password'"
              :background-color="'#d3d3d3'"
              type="password"
              :rules="passwordRule()"
            />
            <v-text-field
              v-model="pwConfirm"
              outlined
              :placeholder="'Password Confirm'"
              :background-color="'#d3d3d3'"
              type="password"
              :rules="passwordConfirmRule(pw)"
            />
          </v-form>
        </div>

        <div class="font-arial" style="margin-top: 30px">
          <v-btn
            style="width: 100%; height: 50px; font-size: 16px"
            class="text-capitalize"
            :color="'#4a80ff'"
            dark
            @click="onCreateAccount"
            :disabled="!isValid"
          >
            Create Account
          </v-btn>
        </div>
        <div class="font-arial" style="margin-top: 10px; font-size: 14px">
          <v-btn
            style="width: 100%; height: 50px; font-size: 16px"
            class="text-capitalize"
            :color="'#707070'"
            dark
            @click="onNavigateLogin"
          >
            Cancel
          </v-btn>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import Tango_login from "@/assets/Tango_login.png";
import logo_3 from "@/assets/icon_3x/Tango_logo.png";

import { userIDRule, passwordConfirmRule, passwordRule, emailRule } from "@/utils";

import { isDuplicateIDAPI, createAccountAPI } from "@/api";

export default {
  data() {
    return {
      Tango_login,
      logo_3,
      userIDRule,
      passwordConfirmRule,
      passwordRule,
      emailRule,
      id: "",
      email: "",
      pw: "",
      pwConfirm: "",
      isValid: false,
      checkID: false
    };
  },

  watch: {
    id() {
      this.isValid = this.$refs.form.validate();
      this.checkID = false;
    },
    email() {
      this.isValid = this.$refs.form.validate();
    },
    pw() {
      this.isValid = this.$refs.form.validate();
    },
    pwConfirm() {
      this.isValid = this.$refs.form.validate();
    }
  },

  methods: {
    /** ID 중복 확인 */
    async onDuplicateIDCheck() {
      const userIDCheck = userIDRule()[0](this.id);
      if (userIDCheck === true) this.checkID = await isDuplicateIDAPI(this.id);
      else {
        this.$swal("ID를 다시 확인해주세요.", "", "error");
      }

      if (this.checkID) {
        this.$swal("ID 중복 확인", "사용가능한 ID 입니다.", "info");
      }
    },

    async onCreateAccount() {
      try {
        if (this.checkID) {
          await createAccountAPI(this.id, this.email, this.pw);
          this.$router.push("login");
        } else {
          this.$swal("ID 중복 검사를 해주세요.", "", "error");
        }
      } catch (err) {
        this.$swal("회원가입에 실패했습니다.", "정보를 다시 확인 후 재 시도 해주세요.", "error");
      }
    },

    /** 회원가입 페이지로 이동 이벤트 */
    onNavigateLogin() {
      this.$router.push("login");
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
  grid-template-rows: 15fr auto 15fr;
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

.v-text-field.v-text-field--enclosed .v-text-field__details {
  padding-top: 4px;
  margin-bottom: 8px;

  height: 15px;
}
</style>
