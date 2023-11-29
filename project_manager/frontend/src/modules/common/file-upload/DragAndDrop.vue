<template>
  <div class="wrapper">
    <div v-if="previewImage" class="d-flex justify-center file-upload-container" :style="{ height: height }">
      <FileListItem @remove="previewRemoveFile">
        <img
          :src="previewImage"
          alt="이미지"
          :style="{
            height: '80%',
            maxHeight: '80%',
            width: 'auto',
            objectFit: 'contain'
          }"
        />
      </FileListItem>
    </div>
    <div
      v-else-if="fileList === null"
      class="file-upload-container file-upload"
      :class="isDragged ? 'dragged' : ''"
      @dragenter="onDragenter"
      @dragover="onDragover"
      @dragleave="onDragleave"
      @drop="onDrop"
      @click="onClick"
    >
      <div class="d-flex flex-column justify-center" :style="{ pointerEvents: isDragged ? 'none' : 'auto' }">
        <v-icon class="mb-3" size="50" color="#c6c6c6"> mdi-file-document-outline </v-icon>
        <slot @dragover="onDragover">
          <div class="notify-text">Drag & Drop Files</div>
        </slot>
      </div>
    </div>

    <div v-else class="d-flex justify-center file-upload-container" :style="{ height: height }">
      <FileListItem v-if="!isImage" @remove="removeFile" :isImage="false">
        <p class="ma-0">{{ fileList.name }}</p>
      </FileListItem>
      <FileListItem v-else @remove="removeFile">
        <img
          :src="fileSrc"
          alt="이미지"
          :style="{
            height: '80%',
            maxHeight: '80%',
            width: 'auto',
            objectFit: 'contain'
          }"
        />
      </FileListItem>
    </div>

    <!-- 파일 업로드 -->
    <input :key="`input-${inputkey}`" type="file" ref="fileInput" class="file-upload-input" @change="onFileChange" />
  </div>
</template>
<script>
import FileListItem from "./FileListItem.vue";
export default {
  components: { FileListItem },
  props: {
    height: {
      required: true
    },
    previewImage: {
      default: null
    },
    errorMsg: {
      default: "이미지 파일을 업로드해 주세요."
    },
    fileType: {
      default: () => ["jpg", "jpeg", "png"]
    },
    isImage: {
      default: true
    }
  },
  data() {
    return {
      isDragged: false,
      fileList: null,
      fileSrc: "",
      mountedItemCount: 0,
      inputkey: 0
    };
  },
  watch: {
    fileList(val) {
      if (val !== null && this.isImage) {
        if (this.fileSrc !== "") URL.revokeObjectURL(this.fileSrc);
        this.fileSrc = URL.createObjectURL(val);
      }
    }
  },

  methods: {
    onClick() {
      if (this.fileList === null) {
        this.$refs.fileInput.click();
      }
    },

    onDragenter() {
      this.isDragged = true;
    },

    onDragleave() {
      this.isDragged = false;
    },

    onDragover(event) {
      event.preventDefault();
    },

    onDrop(event) {
      this.$emit("startUpload");

      if (this.fileList === null) {
        event.preventDefault();

        this.isDragged = false;
        const files = event.dataTransfer.files;
        if (this.fileValidation(files[0])) {
          this.createFile(files[0]);
        }
      }
    },

    onFileChange(event) {
      if (this.fileList === null) {
        event.preventDefault();

        this.isDragged = false;
        const files = event.target.files;
        if (this.fileValidation(files[0])) {
          this.createFile(files[0]);
        }
      }
    },

    createFile(file) {
      this.patchFile(file);
    },

    patchFile(file) {
      this.fileList = file;
      this.$emit("upload", file);
    },

    fileValidation(file) {
      const fileExt = file.type.split("/")[1];
      const type = this.fileType;
      this.isDragged = false;

      if (type.length > 0 && !type.some(q => fileExt.toLowerCase().includes(q.toLowerCase()))) {
        this.$swal("업로드 실패", this.errorMsg, "error");
        return false;
      }

      return true;
    },

    removeFile() {
      this.fileList = null;
      this.isDragged = false;
      URL.revokeObjectURL(this.fileSrc);
      this.fileSrc = "";

      this.inputkey++;
    },

    previewRemoveFile() {
      this.$EventBus.$emit("previewRemove");
      this.removeFile();
    }
  }
};
</script>
<style lang="scss" scoped>
.wrapper {
  width: 100%;
  height: 98%;
  margin: 0 auto;
}
.file-upload {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  border: transparent;
  border-radius: 20px;
  cursor: pointer;
  &.dragged {
    border: 1px dashed powderblue;
    opacity: 0.6;
  }
  &-container {
    position: relative;
    height: 100%;

    margin: 0 auto;
    border: 1px dashed #7fc4fd;
    border-radius: 10px;
  }
  &-input {
    display: none;
  }
}
.delete-btn {
  position: absolute;
  top: 5px;
  right: 5px;
}
</style>
