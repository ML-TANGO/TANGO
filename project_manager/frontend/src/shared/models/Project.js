import { ProjectRequiredColumn } from "@/shared/enums";
import { ProjectType } from "@/shared/consts";

import { getDatasetInfo, getDatasetFolderSize, getDatasetFileCount, updateProjectType } from "@/api";
import Vue from "vue";

export class Project {
  constructor(projectInfo) {
    this.id = projectInfo["id"];
    this.project_name = projectInfo["project_name"];
    this.project_description = projectInfo["project_description"];
    this.create_user = projectInfo["create_user"];
    this.create_date = projectInfo["create_date"];
    this.project_type = projectInfo["project_type"];
    this.target_id = projectInfo["target_id"];
    this.dataset = projectInfo["dataset"];
    this.datasetObject = null;
    this.task_type = projectInfo["task_type"];
    this.learning_type = projectInfo["learning_type"];
    this.weight_file = projectInfo["weight_file"];
    this.autonn_dataset_file = projectInfo["autonn_dataset_file"];
    this.autonn_basemodel = projectInfo["autonn_basemodel"];
    this.nas_type = projectInfo["nas_type"];
    this.deploy_weight_level = projectInfo["deploy_weight_level"];
    this.deploy_precision_level = projectInfo["deploy_precision_level"];
    this.deploy_processing_lib = projectInfo["deploy_processing_lib"];
    this.deploy_user_edit = projectInfo["deploy_user_edit"];
    this.deploy_input_method = projectInfo["deploy_input_method"];
    this.deploy_input_data_path = projectInfo["deploy_input_data_path"];
    this.deploy_output_method = projectInfo["deploy_output_method"];
    this.deploy_input_source = projectInfo["deploy_input_source"];
    this.container = projectInfo["container"];
    this.container_status = projectInfo["container_status"];
    this.last_logs_timestamp = projectInfo["last_logs_timestamp"];
    this.last_log_container = projectInfo["last_log_container"];
    this.current_log = projectInfo["current_log"];
    this.target_info = projectInfo["target_info"];
    this.workflow = projectInfo["workflow"];
  }

  /**
   * Project 설정이 모두 완료 되었는지 확인
   * @returns 설정 완료 : true, else : false
   */
  validation() {
    try {
      for (const column of ProjectRequiredColumn) {
        if (!this[column]) {
          return false;
        }
      }
      return true;
    } catch (err) {
      return false;
    }
  }

  /**
   * dataset load
   */
  async load() {
    if (this.dataset) {
      const res = await getDatasetInfo(this.dataset);
      this.datasetObject = res?.dataset || [];
      if (this.datasetObject) {
        getDatasetFolderSize([this.datasetObject["path"]]).then(res => {
          if (res.datas.length > 0) Vue.set(this.datasetObject, "size", res.datas[0].size);
        });

        getDatasetFileCount([this.datasetObject["path"]]).then(res => {
          if (res.datas.length > 0) Vue.set(this.datasetObject, "file_count", res.datas[0].count);
        });
      }
    }

    if (!this.project_type) {
      await updateProjectType(this.id, ProjectType.MANUAL);
      this.project_type = ProjectType.MANUAL;
    }
  }
}
