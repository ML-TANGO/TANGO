<template>
  <v-dialog v-model="isOpen" width="650px" persistent>
    <v-card>
      <v-card-title style="font-size: 14px">
        <div class="d-flex mr-8">
          <v-btn @click="onUndo" style="min-width: 30px; width: 30px"><v-icon>mdi-chevron-left</v-icon></v-btn>
          <v-btn @click="onRedo" style="min-width: 30px; width: 30px"><v-icon>mdi-chevron-right</v-icon></v-btn>
        </div>
        {{ this.pathHistory?.[this.historyIndex] }}
      </v-card-title>
      <v-card-text>
        <div style="height: 200px; overflow-y: auto" @click="onBgClick">
          <div class="d-flex flex-wrap" style="gap: 8px; overflow-x: hidden">
            <!-- folders -->
            <div v-for="(item, index) in currentPathItems.dirs" :key="`folder-${index}`">
              <v-tooltip bottom>
                <template v-slot:activator="{ on, attrs }">
                  <div
                    class="path_item"
                    :class="item === currentSelected ? 'active' : ''"
                    @dblclick="dbClickFolder($event, item)"
                    v-bind="attrs"
                    v-on="on"
                  >
                    <v-icon size="64">mdi-folder-outline</v-icon>
                    <div class="item_name">{{ item }}</div>
                  </div>
                </template>
                <span style="font-size: 14px">{{ item }}</span>
              </v-tooltip>
            </div>

            <!-- files -->
            <div v-for="(item, index) in currentPathItems.files" :key="`file-${index}`">
              <v-tooltip bottom>
                <template v-slot:activator="{ on, attrs }">
                  <div
                    class="path_item"
                    :class="item === currentSelected ? 'active' : ''"
                    @click="selectFile($event, item, currentPathItems.org_path)"
                    @dblclick="dbClickFile($event, item, currentPathItems.org_path)"
                    v-bind="attrs"
                    v-on="on"
                  >
                    <v-icon size="64">mdi-file-document-outline</v-icon>
                    <div class="item_name">{{ item }}</div>
                  </div>
                </template>
                <span style="font-size: 14px">{{ item }}</span>
              </v-tooltip>
            </div>
          </div>
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer />

        <div class="d-flex" style="gap: 8px">
          <div v-if="currentSelected" class="d-flex align-center">선택 : {{ currentSelected }}</div>
          <v-btn color="tango" dark @click="onSelect" :disabled="!currentSelectedFullPath">선택</v-btn>
          <v-btn @click="onClose">취소</v-btn>
        </div>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>
<script>
import { cloneDeep } from "lodash";
export default {
  props: {
    structure: {
      default: () => []
    }
  },

  data() {
    return {
      isOpen: false,

      currentSelected: "",
      currentSelectedFullPath: "",

      pathHistory: [],
      historyIndex: -1
    };
  },

  computed: {
    // 현재 경로에 맞는 folder와 files를 반환함
    currentPathItems() {
      if (!this.pathHistory?.[this.historyIndex]) return { dirs: [], files: [] };
      else {
        const current = this.structure.find(q => q.path === this.pathHistory[this.historyIndex]);
        const clone = cloneDeep(current);

        return {
          dirs: clone.dirs,
          files: clone.files.filter(q => q.endsWith(".pt")),
          path: clone.path,
          org_path: clone.org_path
        };
      }
    }
  },

  watch: {
    structure: {
      immediate: true,
      handler() {
        if (this.structure.length > 0) {
          this.setCurrentPath(this.structure[0]?.path || "");
        }
      }
    }
  },

  methods: {
    onUndo() {
      if (this.historyIndex - 1 >= 0) {
        this.historyIndex -= 1;
        this.currentSelected = "";
        this.currentSelectedFullPath = "";
      }
    },

    onRedo() {
      if (this.historyIndex + 1 < this.pathHistory.length) {
        this.historyIndex += 1;
        this.currentSelected = "";
        this.currentSelectedFullPath = "";
      }
    },

    setCurrentPath(currentPath) {
      if (currentPath === "") return;
      if (!this.structure.find(q => q.path === currentPath)) return;

      this.pathHistory.splice(this.historyIndex + 1, this.pathHistory.length);
      this.pathHistory.push(currentPath);
      this.historyIndex++;
    },

    onBgClick() {
      this.currentSelected = "";
      this.currentSelectedFullPath = "";
    },

    selectFolder(event) {
      event.preventDefault();
      event.cancelBubble = true;

      //   this.currentSelected = path;
    },
    dbClickFolder(event, path) {
      event.preventDefault();
      event.cancelBubble = true;

      this.setCurrentPath(`${this.pathHistory[this.historyIndex]}/${path}`);
    },

    selectFile(event, fileName, path) {
      event.preventDefault();
      event.cancelBubble = true;

      this.currentSelected = fileName;
      this.currentSelectedFullPath = `${path}/${fileName}`;
    },
    dbClickFile(event, fileName, path) {
      event.preventDefault();
      event.cancelBubble = true;

      this.currentSelected = fileName;
      this.currentSelectedFullPath = `${path}/${fileName}`;
    },

    onSelect() {
      if (!this.currentSelectedFullPath) return;
      this.$emit("select", this.currentSelectedFullPath);
      this.onClose();
    },

    onClose() {
      this.isOpen = false;

      this.currentSelected = "";
      this.currentSelectedFullPath = "";
    }
  }
};
</script>
<style lang="css" scoped>
.path_item {
  display: flex;
  flex-direction: column;

  /* justify-content: center; */
  align-items: center;

  border-radius: 4px;
  width: 68px;
}

.path_item:hover {
  background-color: lightgrey;
}

.path_item.active {
  background-color: #4a98fb;
}

.path_item .item_name {
  width: 68px;
  padding: 0px 4px;
  font-size: 12px;

  color: black;

  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;

  /* display: -webkit-box; */
  /* -webkit-line-clamp: 2; */
  /* -webkit-box-orient: vertical; */

  text-align: center;
}
</style>
