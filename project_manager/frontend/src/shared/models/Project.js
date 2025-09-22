import { ProjectRequiredColumn, TaskType } from "@/shared/enums";
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
    this.version = projectInfo?.["version"] || 0;
  }

  /**
   * Project 설정이 모두 완료 되었는지 확인
   * @returns 설정 완료 : true, else : false
   */
  validation() {
    try {
      // Segmentation 프로젝트는 별도의 검증 규칙 적용
      if (this.task_type === TaskType.SEGMENTATION) {
        return this.validateSegmentationProject();
      }

      for (const column of ProjectRequiredColumn) {
        if (!this[column]) {
          return false;
        }
      }

      if (this.task_type !== TaskType.CHAT && (!this.dataset || this.dataset === "")) return false;

      return true;
    } catch (err) {
      return false;
    }
  }

  /**
   * Segmentation 프로젝트 전용 검증 로직
   * @returns 검증 성공 : true, else : false
   */
  validateSegmentationProject() {
    try {
      // Segmentation 프로젝트에 필수인 필드들만 검증
      const segmentationRequiredFields = [
        "task_type",
        "deploy_weight_level",
        "deploy_precision_level",
        "deploy_user_edit",
        "deploy_output_method"
      ];

      for (const field of segmentationRequiredFields) {
        if (!this[field]) {
          console.log(`Segmentation validation failed: ${field} is missing`);
          return false;
        }
      }

      // Segmentation은 target_id와 dataset이 없어도 유효함
      console.log("Segmentation project validation passed");
      return true;
    } catch (err) {
      console.error("Segmentation validation error:", err);
      return false;
    }
  }

  /**
   * dataset load
   */
  async load() {
    // Segmentation 프로젝트는 dataset 로딩 생략
    if (this.task_type === TaskType.SEGMENTATION) {
      console.log("Segmentation project: skipping dataset load");
      if (!this.project_type) {
        await updateProjectType(this.id, ProjectType.AUTO); // Segmentation은 AUTO 타입으로 설정
        this.project_type = ProjectType.AUTO;
      }
      return;
    }

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
