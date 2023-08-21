<template>
  <div class="d-flex flex-column justify-space-between" style="height: 88%">
    <div>
      <p class="text-h5 mb-10" style="color: #4a80ff">Target Informations</p>
      <v-text-field :value="targetName" outlined dense label="Target Name" @change="onChange" />
      <DragAndDrop :height="'224px'" @upload="onUpload" :previewImage="image" />
    </div>
    <div class="d-flex justify-end">
      <v-btn class="ma-0 pa-0" text style="color: #4a80ff" @click="next"> NEXT </v-btn>
    </div>
  </div>
</template>
<script>
import { mapState } from "vuex";
import { TargetNamespace } from "@/store/modules/targetStore";
import DragAndDrop from "@/modules/common/file-upload/DragAndDrop.vue";
import imageCompression from "browser-image-compression";
export default {
  components: { DragAndDrop },
  data() {
    return {
      targetName: "",
      image: ""
    };
  },

  computed: {
    ...mapState(TargetNamespace, ["target"])
  },

  mounted() {
    this.targetName = this.target?.name;
    this.image = this.target?.image;

    this.$EventBus.$on("previewRemove", this.previewRemoveFile);
  },

  methods: {
    next() {
      if (this.targetName === "" || !this.targetName) {
        this.$swal("Target", "Target 이름을 입력해 주세요.", "error");
        return;
      } else if (this.image === "" || !this.image) {
        this.$swal("Target", "Target 이미지를 입력해 주세요.", "error");
        return;
      }
      this.$emit("next", { name: this.targetName, image: this.image });
    },

    async onUpload(file) {
      this.$swal.fire({
        title: "Please Wait....",
        allowOutsideClick: false,
        allowEscapekey: false,
        allowEnterkey: false,
        showConfimButton: false,
        didOpen: () => {
          this.$swal.showLoading();
        }
      });

      const options = {
        maxSizeMB: 0.2,
        maxWidthOrHeight: 720,
        useWebWorker: true
      };

      const compressFile = await imageCompression(file, options);
      const reader = new FileReader();
      reader.onload = () => {
        this.image = reader.result;
        this.$swal.close();
      };
      reader.readAsDataURL(compressFile);
    },

    onChange(name) {
      this.targetName = name;
    },

    previewRemoveFile() {
      this.image = null;
    }
  }
};
</script>
<style lang="scss" scoped></style>
