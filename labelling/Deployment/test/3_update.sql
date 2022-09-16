ALTER TABLE tab_featureinfo ADD `COLUMN_IDX` int
(11) NOT NULL DEFAULT 0;

ALTER TABLE TAB_TRAINFEATURE ADD `COLUMN_IDX` int
(11) NULL DEFAULT 0;




ALTER TABLE output_type ADD `TARGET` varchar
(10) NOT NULL DEFAULT 'A';

ALTER TABLE prj_report ADD `PRC_TIME` double DEFAULT NULL;

ALTER TABLE prj_report ADD `REQ_DATA` longtext DEFAULT NULL;

ALTER TABLE prj_report ADD `RESULT_DATA` longtext DEFAULT NULL;

ALTER TABLE prj_report ADD `RS_STATUS` int
(11) DEFAULT 0;

ALTER TABLE prj_report ADD `OUT_CD` int
(11) DEFAULT NULL;

ALTER TABLE prj_report ADD `REQ_NO` int
(11) DEFAULT 0;


ALTER TABLE TRAIN_LOG MODIFY COLUMN AI_LOSS int
(11) DEFAULT 0 NULL;


ALTER TABLE TRAIN_LOG MODIFY COLUMN GPU_RATE float
(11) DEFAULT 0 NULL;

ALTER TABLE TRAIN_LOG MODIFY COLUMN AI_ACC float
(11) DEFAULT 0 NULL;

ALTER TABLE TRAIN_LOG MODIFY COLUMN AI_VAL_LOSS float
(11) DEFAULT 0 NULL;

ALTER TABLE TRAIN_LOG MODIFY COLUMN AI_VAL_ACC float
(11) DEFAULT 0 NULL;

ALTER TABLE TRAIN_LOG DROP PRIMARY KEY;

ALTER TABLE TRAIN_LOG ADD CONSTRAINT TRAIN_LOG_PK PRIMARY KEY (AI_CD,EPOCH,MDL_IDX);




ALTER TABLE train_modelinfo MODIFY COLUMN LAGACY_MDL_NM VARCHAR
(64) CHARACTER
SET utf8mb4
COLLATE utf8mb4_general_ci NULL;

ALTER TABLE train_modelinfo MODIFY COLUMN LAGACY_MDL_NM VARCHAR
(64) CHARACTER
SET utf8mb4
COLLATE utf8mb4_general_ci NULL;

ALTER TABLE train_modelinfo MODIFY COLUMN MDL_IDX INT DEFAULT 0 CHARACTER
SET utf8mb4
COLLATE utf8mb4_general_ci NULL;

ALTER TABLE train_modelinfo MODIFY COLUMN MDL_ALIAS VARCHAR
(128) CHARACTER
SET utf8mb4
COLLATE utf8mb4_general_ci NULL;

ALTER TABLE train_modelinfo MODIFY COLUMN PARAM VARCHAR
(4000) CHARACTER
SET utf8mb4
COLLATE utf8mb4_general_ci NULL;

ALTER TABLE users ADD `PREV_PW` VARCHAR
(256) NOT NULL;

ALTER TABLE users ADD `
USE` CHAR
(1) NOT NULL;

ALTER TABLE users ADD `LOGIN_CNT` VARCHAR
(10) NOT NULL;

ALTER TABLE users ADD `CRN_DTM` datetime NOT NULL;

ALTER TABLE users ADD `UPT_DTM` datetime NOT NULL;

set global innodb_file_format=Barracuda;

set global innodb_file_per_table=ON;

set global innodb_large_prefix=ON;

set global innodb_default_row_format=dynamic;

ALTER TABLE DATA_ELEMENT MODIFY COLUMN FILE_NAME varchar(256) NOT NULL;

ALTER TABLE FILE_LIST MODIFY COLUMN FILE_NAME varchar(256) NOT NULL;

ALTER TABLE FILE_LIST MODIFY COLUMN FILE_PATH varchar(800) NOT NULL;

ALTER TABLE TAB_ANALYSIS MODIFY COLUMN SAMPLES longtext;

ALTER TABLE INPUT_SOURCE ADD SERVICE_AUTH varchar(256) DEFAULT NULL NULL;

UPDATE BASE_MODELS SET NETWORK_PATH = '/var/appdata/models/classification/efficientnet/' where  NETWORK_NAME = 'efficientnet';

ALTER TABLE TRAIN_LOG ADD REMANING_TIME FLOAT DEFAULT 0 NULL;

ALTER TABLE tab_trainset ADD FE_MODE varchar(10) DEFAULT NULL NULL;


INSERT INTO BASE_MODELS
	(NETWORK_INFO,OBJECT_TYPE,DATA_TYPE,NETWORK_NAME,NETWORK_PATH,PIRIORITY)
VALUES
('Classification Default Model', 'C', 'I', 'vgg16', '/var/appdata/models/classification/vgg', 2),
('Classification Default Model', 'C', 'I', 'vgg19', '/var/appdata/models/classification/vgg', 2),
('Classification Default Model', 'C', 'I', 'resnet101', '/var/appdata/models/classification/resnet', 2),
('Classification Default Model', 'C', 'I', 'resnet101v2', '/var/appdata/models/classification/resnet', 2),
('Classification Default Model', 'C', 'I', 'resnet152', '/var/appdata/models/classification/resnet', 2),
('Classification Default Model', 'C', 'I', 'resnet152v2', '/var/appdata/models/classification/resnet', 2),
('Classification Default Model', 'C', 'I', 'resnet50', '/var/appdata/models/classification/resnet', 2),
('Classification Default Model', 'C', 'I', 'resnet50v2', '/var/appdata/models/classification/resnet', 2),
('Classification Default Model', 'C', 'I', 'inceptionv3', '/var/appdata/models/classification/inception', 2),
('Classification Default Model', 'C', 'I', 'inceptionresnetv2', '/var/appdata/models/classification/inception', 2),
('Classification Default Model', 'C', 'I', 'mobilenet', '/var/appdata/models/classification/mobilenet', 2),
('Classification Default Model', 'C', 'I', 'mobilenetv2', '/var/appdata/models/classification/mobilenet', 2);

INSERT INTO BASE_MODELS
	(NETWORK_INFO,OBJECT_TYPE,DATA_TYPE,NETWORK_NAME,NETWORK_PATH,PIRIORITY)
VALUES
('Segmentation Default Model', 'S', 'I', 'U-NET', '/var/appdata/models/segmentation/unet', 2),
('Segmentation Default Model', 'S', 'V', 'U-NET', '/var/appdata/models/segmentation/unet', 2);