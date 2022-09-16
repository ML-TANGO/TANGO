INSERT INTO WEDA.BASE_MODELS
	(NETWORK_INFO,OBJECT_TYPE,DATA_TYPE,NETWORK_NAME,NETWORK_PATH,PIRIORITY)
VALUES
	('Classification Auto Model', 'C', 'I', 'AUTOKERAS', '/var/appdata/models/classification/autokeras', 1),
	('Classification Auto Model', 'C', 'V', 'AUTOKERAS', '/var/appdata/models/classification/autokeras', 1),
	('Classification Default Model', 'C', 'I', 'efficientnet', '/var/appdata/models/classification/efficientnet/', 2),
	('Classification Default Model', 'C', 'V', 'efficientnet', '/var/appdata/models/classification/efficientnet/', 2),
	('Detection Auto Model', 'D', 'I', 'EFFICIENTDET', '/var/appdata/models/detection/efficientdet', 1),
	('Detection Auto Model', 'D', 'V', 'EFFICIENTDET', '/var/appdata/models/detection/efficientdet', 1),
	('Detection Default Model', 'D', 'I', 'YOLOV3', '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 2),
	('Detection Default Model', 'D', 'V', 'YOLOV3', '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 2),
  ('Detection Default Model', 'D', 'I', 'YOLOV4', '/var/appdata/models/detection/yolo/yolov4_checkpoints/', 2),
  ('Detection Default Model', 'D', 'V', 'YOLOV4', '/var/appdata/models/detection/yolo/yolov4_checkpoints/', 2),
	('Segmentation Auto Model', 'S', 'I', 'EFFICIENTDET-SEG', '/var/appdata/models/segmentation/efficientdet-seg', 1),
	('Segmentation Auto Model', 'S', 'V', 'EFFICIENTDET-SEG', '/var/appdata/models/segmentation/efficientdet-seg', 1),
	('Segmentation Default Model', 'S', 'I', 'DEEP-LAB', '/var/appdata/models/segmentation/deeplab/', 2),
	('Segmentation Default Model', 'S', 'V', 'DEEP-LAB', '/var/appdata/models/segmentation/deeplab/', 2),
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
  ('Classification Default Model', 'C', 'I', 'mobilenetv2', '/var/appdata/models/classification/mobilenet', 2),
  ('Segmentation Default Model', 'S', 'I', 'U-NET', '/var/appdata/models/segmentation/unet', 2),
  ('Segmentation Default Model', 'S', 'V', 'U-NET', '/var/appdata/models/segmentation/unet', 2);

INSERT INTO WEDA.CODE_INFO
	(CODE_TYPE,DP_NAME,DB_NAME)
VALUES
	(1, 'ELU', 'elu'),
	(1, 'EXPONENTIAL', 'exponential'),
	(1, 'RELU', 'relu'),
	(1, 'SELU', 'selu'),
	(1, 'Sigmoid', 'sigmoid'),
	(1, 'SoftMax', 'softmax'),
	(1, 'SOFTPLUS', 'softplus'),
	(1, 'SOFTSIGN', 'softsign'),
	(1, 'TANH', 'tanh'),
	(2, 'ADADELTA', 'adadelta'),
	(2, 'ADAGRAD', 'adagrad'),
	(2, 'ADAM', 'adam'),
	(2, 'ADAMAX', 'adamax'),
	(2, 'FTRL', 'ftrl'),
	(2, 'NADAM', 'nadam'),
	(2, 'RMSPROPS', 'rmsprops'),
	(2, 'SGD', 'sgd'),
	(3, 'binary_crossentropy', 'binary_crossentropy'),
	(3, 'categorical_crossentropy', 'categorical_crossentropy'),
	(3, 'MSE', 'mean_squared_error');
INSERT INTO WEDA.CODE_INFO
	(CODE_TYPE,DP_NAME,DB_NAME)
VALUES
	(3, 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy');

INSERT INTO WEDA.CODE_KIND
	(CODE_TYPE,CODE_KIND,CODE_LOCATION)
VALUES
	(1, 'ACTIVATE_FUNC', 'NEW_AI'),
	(2, 'OPTIMIZER', 'NEW_AI'),
	(3, 'LOSS_FUNC', 'NEW_AI');

INSERT INTO WEDA.COMPONENT_DEF
	(COMP_TYPE, IS_TYPE, DEF_TXT, COMPONENT_NAME)
VALUES
	(1, 'I', '{ "name": "ImageViewer", "type": "I", "w": 12, "h": 8, "minW": 10, "isUse": false }', 'ImageViewer'),
	(2, 'I', '{ "name": "ImageBarChart", "type": "I", "w": 8, "h": 5, "isUse": false }', 'ImageBarChart'),
	(3, 'I', '{ "name": "ImagePieChart", "type": "I", "w": 5, "h": 5, "isUse": false }', 'ImagePieChart'),
	(4, 'I', '{ "name": "ImageCount", "type": "I", "w": 4, "h": 3, "minW": 4, "minH": 3, "isUse": false }', 'ImageCount'),
	(5, 'I', '{ "name": "ImageLineChart", "type": "I", "w": 5, "h": 5, "isUse": false }', 'ImageLineChart'),
	(6, 'V', '{ "name": "VideoViewer", "type": "V", "w": 12, "h": 8, "minW": 10, "isUse": false }', 'VideoViewer'),
	(7, 'V', '{ "name": "VideoBarChart", "type": "V", "w": 8, "h": 5, "isUse": false }', 'VideoBarChart'),
	(8, 'V', '{ "name": "VideoPieChart", "type": "V", "w": 5, "h": 5, "isUse": false }', 'VideoPieChart'),
	(9, 'V', '{ "name": "VideoLineChart", "type": "V", "w": 5, "h": 5, "isUse": false }', 'VideoLineChart'),
	(10, 'R', '{ "name": "RealTimeViewer", "type": "R", "w": 12, "h": 8, "minW": 10, "isUse": false }', 'RealTimeViewer'),
	(11, 'R', '{ "name": "TimeLine", "type": "R", "w": 24, "h": 4, "minW": 4, "minH": 2, "isUse": false }', 'TimeLine'),
	(12, 'R', '{ "name": "PredictViewer", "type": "R", "w": 12, "h": 8, "minW": 10, "isUse": false }', 'PredictViewer'),
	(13, 'R', '{ "name": "RealTimeBubbleChart", "type": "R", "w": 10, "h": 10, "minW": 10, "minH": 7, "isUse": false }', 'RealTimeBubbleChart'),
	(14, 'R', '{ "name": "RealTimeBarChart", "type": "R", "w": 8, "h": 5, "isUse": false }', 'RealTimeBarChart'),
	(15, 'R', '{ "name": "RealTimePieChart", "type": "R", "w": 5, "h": 5, "isUse": false}', 'RealTimePieChart'),
	(16, 'R', '{ "name": "RealTimeLineChart", "type": "R", "w": 5, "h": 5, "isUse": false }', 'RealTimeLineChart'),
	(17, 'R', '{ "name": "RealTimeStatusChart", "type": "R", "w": 5, "h": 5, "isUse": false }', 'RealTimeStatusChart'),
	(18, 'R', '{ "name": "ClassificationBoard", "type": "R", "objectType" : "C" ,"w": 4, "h": 3, "minW":4, "minH": 3, "isUse": false }', 'ClassificationBoard'),
	(19, 'T', '{ "name": "ResultBoard", "type": "T", "w": 24, "h": 8, "minW":6, "minH": 6, "isUse": false }', 'ResultBoard'),
	(20, 'T', '{ "name": "RealTimeTransactionChart", "type": "T", "w": 5, "h": 5, "isUse": false }', 'RealTimeTransactionChart'),
	(21, 'T', '{ "name": "InformationBoard", "type": "T", "w": 4, "h": 3, "minW":4, "minH": 3, "isUse": false }', 'InformationBoard'),
	(22, 'T', '{ "name": "TimeLine", "type": "T", "w": 24, "h": 4, "minW": 4, "minH": 2, "isUse": false }', 'TimeLine');

INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(1,0,1,'Person','C'),
(2,0,2,'animal','C'),
(3,0,3,'objects','C'),
(4,0,4,'plant','C'),
(5,0,5,'food','C'),
(6,0,6,'terrain','C'),
(7,0,7,'Etc','C'),
(8,0,8,'Person','D'),
(9,0,9,'animal','D'),
(10,0,10,'objects','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(11,0,11,'plant','D'),
(12,0,12,'food','D'),
(13,0,13,'terrain','D'),
(14,0,14,'Etc','D'),
(15,0,15,'Person','S'),
(16,0,16,'animal','S'),
(17,0,17,'objects','S'),
(18,0,18,'plant','S'),
(19,0,19,'food','S'),
(20,0,20,'terrain','S');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(21,0,21,'Etc','S'),
(22,1,2,'dog','C'),
(23,1,2,'mammalia','C'),
(24,1,2,'cat','C'),
(25,1,7,'Rodent','C'),
(26,1,2,'Aquatic mammals','C'),
(27,1,7,'Primates','C'),
(28,1,4,'pool','C'),
(29,1,7,'Etc','C'),
(30,1,3,'accessory','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(31,1,3,'Vehicle','C'),
(32,1,7,'sports','C'),
(33,1,3,'instrument','C'),
(34,1,7,'insect','C'),
(35,1,2,'Aquatic animals','C'),
(36,1,2,'Mollusk','C'),
(37,1,3,'Electronics','C'),
(38,1,5,'fruit','C'),
(39,1,7,'amphibia','C'),
(40,1,3,'furniture','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(41,1,3,'Things outside','C'),
(42,1,1,'body','C'),
(43,1,4,'Flower','C'),
(44,1,6,'terrain','C'),
(45,1,3,'tool','C'),
(46,1,3,'kitchen utensils','C'),
(47,1,2,'bird','C'),
(48,1,7,'fish','C'),
(49,1,7,'reptile','C'),
(50,1,3,'Home Appliances','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(51,1,2,'Arthropod','C'),
(52,1,1,'Person','C'),
(53,1,3,'Kitchen appliances','C'),
(54,1,3,'Home Appliances','C'),
(55,1,3,'Kitchen equipment','C'),
(56,1,3,'building','C'),
(57,1,3,'Structure outside','C'),
(58,1,5,'vegetable','C'),
(59,1,3,'dress','C'),
(60,1,3,'shoes','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(61,1,8,'Person','D'),
(62,1,10,'Vehicle','D'),
(63,1,10,'Things outside','D'),
(64,1,9,'bird','D'),
(65,1,9,'cat','D'),
(66,1,9,'dog','D'),
(67,1,9,'horse','D'),
(68,1,9,'sheep','D'),
(69,1,9,'Cow','D'),
(70,1,9,'elephant','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(71,1,9,'bear','D'),
(72,1,9,'zebra','D'),
(73,1,10,'accessory','D'),
(74,1,14,'sports','D'),
(75,1,10,'Baseball bat','D'),
(76,1,10,'Baseball glove','D'),
(77,1,10,'Skateboard','D'),
(78,1,10,'kitchen utensils','D'),
(79,1,12,'food','D'),
(80,1,10,'furniture','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(81,1,10,'Indoor plants','D'),
(82,1,10,'Home Appliances','D'),
(83,1,10,'Kitchen appliances','D'),
(84,1,10,'Objects in the house','D'),
(85,1,9,'giraffe','D'),
(86,1,17,'Vehicle','S'),
(87,1,16,'bird','S'),
(88,1,17,'kitchen utensils','S'),
(89,1,16,'cat','S'),
(90,1,17,'furniture','S');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(91,1,16,'Cow','S'),
(92,1,16,'dog','S'),
(93,1,16,'horse','S'),
(94,1,15,'Person','S'),
(95,1,16,'sheep','S'),
(96,1,17,'Home Appliances','S'),
(97,2,22,'More than 50%','C'),
(98,2,23,'More than 50%','C'),
(99,2,24,'More than 50%','C'),
(100,2,25,'More than 50%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(101,2,26,'More than 50%','C'),
(102,2,27,'More than 50%','C'),
(103,2,28,'More than 50%','C'),
(104,2,29,'More than 50%','C'),
(105,2,30,'More than 50%','C'),
(106,2,31,'More than 50%','C'),
(107,2,32,'More than 50%','C'),
(108,2,33,'More than 50%','C'),
(109,2,34,'More than 50%','C'),
(110,2,35,'More than 50%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(111,2,36,'More than 50%','C'),
(112,2,37,'More than 50%','C'),
(113,2,38,'More than 50%','C'),
(114,2,39,'More than 50%','C'),
(115,2,40,'More than 50%','C'),
(116,2,41,'More than 50%','C'),
(117,2,42,'More than 50%','C'),
(118,2,43,'More than 50%','C'),
(119,2,44,'More than 50%','C'),
(120,2,45,'More than 50%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(121,2,46,'More than 50%','C'),
(122,2,47,'More than 50%','C'),
(123,2,48,'More than 50%','C'),
(124,2,49,'More than 50%','C'),
(125,2,50,'More than 50%','C'),
(126,2,51,'More than 50%','C'),
(127,2,52,'More than 50%','C'),
(128,2,53,'More than 50%','C'),
(129,2,54,'More than 50%','C'),
(130,2,55,'More than 50%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(131,2,56,'More than 50%','C'),
(132,2,57,'More than 50%','C'),
(133,2,58,'More than 50%','C'),
(134,2,59,'More than 50%','C'),
(135,2,60,'More than 50%','C'),
(136,2,61,'More than 50%','D'),
(137,2,62,'More than 50%','D'),
(138,2,63,'More than 50%','D'),
(139,2,64,'More than 50%','D'),
(140,2,65,'More than 50%','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(141,2,66,'More than 50%','D'),
(142,2,67,'More than 50%','D'),
(143,2,68,'More than 50%','D'),
(144,2,69,'More than 50%','D'),
(145,2,70,'More than 50%','D'),
(146,2,71,'More than 50%','D'),
(147,2,72,'More than 50%','D'),
(148,2,73,'More than 50%','D'),
(149,2,74,'More than 50%','D'),
(150,2,75,'More than 50%','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(151,2,76,'More than 50%','D'),
(152,2,77,'More than 50%','D'),
(153,2,78,'More than 50%','D'),
(154,2,79,'More than 50%','D'),
(155,2,80,'More than 50%','D'),
(156,2,81,'More than 50%','D'),
(157,2,82,'More than 50%','D'),
(158,2,83,'More than 50%','D'),
(159,2,84,'More than 50%','D'),
(160,2,85,'More than 50%','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(161,2,86,'More than 50%','S'),
(162,2,87,'More than 50%','S'),
(163,2,88,'More than 50%','S'),
(164,2,89,'More than 50%','S'),
(165,2,90,'More than 50%','S'),
(166,2,91,'More than 50%','S'),
(167,2,92,'More than 50%','S'),
(168,2,93,'More than 50%','S'),
(169,2,94,'More than 50%','S'),
(170,2,95,'More than 50%','S');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(171,2,96,'More than 50%','S'),
(172,2,22,'More than 90%','C'),
(173,2,23,'More than 90%','C'),
(174,2,24,'More than 90%','C'),
(175,2,25,'More than 90%','C'),
(176,2,26,'More than 90%','C'),
(177,2,27,'More than 90%','C'),
(178,2,28,'More than 90%','C'),
(179,2,29,'More than 90%','C'),
(180,2,30,'More than 90%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(181,2,31,'More than 90%','C'),
(182,2,32,'More than 90%','C'),
(183,2,33,'More than 90%','C'),
(184,2,34,'More than 90%','C'),
(185,2,35,'More than 90%','C'),
(186,2,36,'More than 90%','C'),
(187,2,37,'More than 90%','C'),
(188,2,38,'More than 90%','C'),
(189,2,39,'More than 90%','C'),
(190,2,40,'More than 90%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(191,2,41,'More than 90%','C'),
(192,2,42,'More than 90%','C'),
(193,2,43,'More than 90%','C'),
(194,2,44,'More than 90%','C'),
(195,2,45,'More than 90%','C'),
(196,2,46,'More than 90%','C'),
(197,2,47,'More than 90%','C'),
(198,2,48,'More than 90%','C'),
(199,2,49,'More than 90%','C'),
(200,2,50,'More than 90%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(201,2,51,'More than 90%','C'),
(202,2,52,'More than 90%','C'),
(203,2,53,'More than 90%','C'),
(204,2,54,'More than 90%','C'),
(205,2,55,'More than 90%','C'),
(206,2,56,'More than 90%','C'),
(207,2,57,'More than 90%','C'),
(208,2,58,'More than 90%','C'),
(209,2,59,'More than 90%','C'),
(210,2,60,'More than 90%','C');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(211,2,61,'More than 90%','D'),
(212,2,62,'More than 90%','D'),
(213,2,63,'More than 90%','D'),
(214,2,64,'More than 90%','D'),
(215,2,65,'More than 90%','D'),
(216,2,66,'More than 90%','D'),
(217,2,67,'More than 90%','D'),
(218,2,68,'More than 90%','D'),
(219,2,69,'More than 90%','D'),
(220,2,70,'More than 90%','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(221,2,71,'More than 90%','D'),
(222,2,72,'More than 90%','D'),
(223,2,73,'More than 90%','D'),
(224,2,74,'More than 90%','D'),
(225,2,75,'More than 90%','D'),
(226,2,76,'More than 90%','D'),
(227,2,77,'More than 90%','D'),
(228,2,78,'More than 90%','D'),
(229,2,79,'More than 90%','D'),
(230,2,80,'More than 90%','D');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(231,2,81,'More than 90%','D'),
(232,2,82,'More than 90%','D'),
(233,2,83,'More than 90%','D'),
(234,2,84,'More than 90%','D'),
(235,2,85,'More than 90%','D'),
(236,2,86,'More than 90%','S'),
(237,2,87,'More than 90%','S'),
(238,2,88,'More than 90%','S'),
(239,2,89,'More than 90%','S'),
(240,2,90,'More than 90%','S');
INSERT INTO WEDA.DATA_CATEGORY
	(CATEGORY_SEQ,`DEPTH`,PARENTS_SEQ,CATEGORY_NAME,OBJECT_TYPE
) VALUES
(241,2,91,'More than 90%','S'),
(242,2,92,'More than 90%','S'),
(243,2,93,'More than 90%','S'),
(244,2,94,'More than 90%','S'),
(245,2,95,'More than 90%','S'),
(246,2,96,'More than 90%','S');

INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'person', 'D', 'I', 'YOLOV3', 'mscoco', '8', '8', '136', 'person'),
	(2, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bicycle', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'bicycle'),
	(3, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'car', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'car'),
	(4, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'motorbike', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'motorbike'),
	(5, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'aeroplane', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'aeroplane'),
	(6, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bus', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'bus'),
	(7, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'train', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'train'),
	(8, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'truck', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'truck'),
	(9, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'boat', 'D', 'I', 'YOLOV3', 'mscoco', '10', '62', '136', 'boat'),
	(10, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'traffic light', 'D', 'I', 'YOLOV3', 'mscoco', '10', '63', '136', 'traffic light');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(11, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'fire hydrant', 'D', 'I', 'YOLOV3', 'mscoco', '10', '63', '136', 'fire hydrant'),
	(12, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'stop sign', 'D', 'I', 'YOLOV3', 'mscoco', '10', '63', '136', 'stop sign'),
	(13, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'parking meter', 'D', 'I', 'YOLOV3', 'mscoco', '10', '63', '136', 'parking meter'),
	(14, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bird', 'D', 'I', 'YOLOV3', 'mscoco', '9', '64', '136', 'bird'),
	(15, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'cat', 'D', 'I', 'YOLOV3', 'mscoco', '9', '65', '136', 'cat'),
	(16, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'dog', 'D', 'I', 'YOLOV3', 'mscoco', '9', '66', '136', 'dog'),
	(17, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'horse', 'D', 'I', 'YOLOV3', 'mscoco', '9', '67', '136', 'horse'),
	(18, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'sheep', 'D', 'I', 'YOLOV3', 'mscoco', '9', '68', '136', 'sheep'),
	(19, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'cow', 'D', 'I', 'YOLOV3', 'mscoco', '9', '69', '136', 'cow'),
	(20, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'elephant', 'D', 'I', 'YOLOV3', 'mscoco', '9', '70', '136', 'elephant');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(21, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bear', 'D', 'I', 'YOLOV3', 'mscoco', '9', '71', '136', 'bear'),
	(22, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'zebra', 'D', 'I', 'YOLOV3', 'mscoco', '9', '72', '136', 'zebra'),
	(23, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'backpack', 'D', 'I', 'YOLOV3', 'mscoco', '10', '73', '136', 'backpack'),
	(24, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'umbrella', 'D', 'I', 'YOLOV3', 'mscoco', '10', '73', '136', 'umbrella'),
	(25, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'handbag', 'D', 'I', 'YOLOV3', 'mscoco', '10', '73', '136', 'handbag'),
	(26, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'tie', 'D', 'I', 'YOLOV3', 'mscoco', '10', '73', '136', 'tie'),
	(27, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'suitcase', 'D', 'I', 'YOLOV3', 'mscoco', '10', '73', '136', 'suitcase'),
	(28, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'frisbee', 'D', 'I', 'YOLOV3', 'mscoco', '10', '74', '136', 'frisbee'),
	(29, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'skis', 'D', 'I', 'YOLOV3', 'mscoco', '10', '74', '136', 'skis'),
	(30, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'snowboard', 'D', 'I', 'YOLOV3', 'mscoco', '10', '74', '136', 'snowboard');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(31, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'sports ball', 'D', 'I', 'YOLOV3', 'mscoco', '10', '74', '136', 'sports ball'),
	(32, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'kite', 'D', 'I', 'YOLOV3', 'mscoco', '10', '74', '136', 'kite'),
	(33, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'baseball bat', 'D', 'I', 'YOLOV3', 'mscoco', '14', '74', '136', 'baseball bat'),
	(34, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'baseball glove', 'D', 'I', 'YOLOV3', 'mscoco', '14', '74', '136', 'baseball glove'),
	(35, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'skateboard', 'D', 'I', 'YOLOV3', 'mscoco', '14', '74', '136', 'skateboard'),
	(36, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'surfboard', 'D', 'I', 'YOLOV3', 'mscoco', '10', '74', '136', 'surfboard'),
	(37, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'tennis racket', 'D', 'I', 'YOLOV3', 'mscoco', '10', '74', '136', 'tennis racket'),
	(38, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bottle', 'D', 'I', 'YOLOV3', 'mscoco', '10', '78', '136', 'bottle'),
	(39, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'wine glass', 'D', 'I', 'YOLOV3', 'mscoco', '10', '78', '136', 'wine glass'),
	(40, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'cup', 'D', 'I', 'YOLOV3', 'mscoco', '10', '78', '136', 'cup');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(41, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'fork', 'D', 'I', 'YOLOV3', 'mscoco', '10', '78', '136', 'fork'),
	(42, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'knife', 'D', 'I', 'YOLOV3', 'mscoco', '10', '78', '136', 'knife'),
	(43, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'spoon', 'D', 'I', 'YOLOV3', 'mscoco', '10', '78', '136', 'spoon'),
	(44, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bowl', 'D', 'I', 'YOLOV3', 'mscoco', '10', '78', '136', 'bowl'),
	(45, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'banana', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'banana'),
	(46, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'sandwich', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'sandwich'),
	(47, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'orange', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'orange'),
	(48, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'broccoli', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'broccoli'),
	(49, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'carrot', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'carrot'),
	(50, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'hot dog', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'hot dog');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(51, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'pizza', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'pizza'),
	(52, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'donut', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'donut'),
	(53, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'cake', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'cake'),
	(54, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'chair', 'D', 'I', 'YOLOV3', 'mscoco', '10', '80', '136', 'chair'),
	(55, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'sofa', 'D', 'I', 'YOLOV3', 'mscoco', '10', '80', '136', 'sofa'),
	(56, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'pottedplant', 'D', 'I', 'YOLOV3', 'mscoco', '10', '81', '136', 'pottedplant'),
	(57, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bed', 'D', 'I', 'YOLOV3', 'mscoco', '10', '80', '136', 'bed'),
	(58, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'diningtable', 'D', 'I', 'YOLOV3', 'mscoco', '10', '80', '136', 'diningtable'),
	(59, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'toilet', 'D', 'I', 'YOLOV3', 'mscoco', '10', '80', '136', 'toilet'),
	(60, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'tvmonitor', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'tvmonitor');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(61, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'laptop', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'laptop'),
	(62, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'mouse', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'mouse'),
	(63, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'remote', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'remote'),
	(64, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'keyboard', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'keyboard'),
	(65, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'cell phone', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'cell phone'),
	(66, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'microwave', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'microwave'),
	(67, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'oven', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'oven'),
	(68, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'toaster', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'toaster'),
	(69, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'sink', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'sink'),
	(70, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'refrigerator', 'D', 'I', 'YOLOV3', 'mscoco', '10', '82', '136', 'refrigerator');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(71, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'book', 'D', 'I', 'YOLOV3', 'mscoco', '10', '84', '136', 'book'),
	(72, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'clock', 'D', 'I', 'YOLOV3', 'mscoco', '10', '84', '136', 'clock'),
	(73, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'vase', 'D', 'I', 'YOLOV3', 'mscoco', '10', '84', '136', 'vase'),
	(74, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'scissors', 'D', 'I', 'YOLOV3', 'mscoco', '10', '84', '136', 'scissors'),
	(75, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'teddy bear', 'D', 'I', 'YOLOV3', 'mscoco', '10', '84', '136', 'teddy bear'),
	(76, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'hair drier', 'D', 'I', 'YOLOV3', 'mscoco', '10', '84', '136', 'hair drier'),
	(77, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'toothbrush', 'D', 'I', 'YOLOV3', 'mscoco', '10', '84', '136', 'toothbrush'),
	(78, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'person', 'D', 'I', 'tiny_yolo', 'mscoco', '8', '8', '136', 'person'),
	(79, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bicycle', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'bicycle'),
	(80, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'car', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'car');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(81, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'motorbike', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'motorbike'),
	(82, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'aeroplane', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'aeroplane'),
	(83, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bus', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'bus'),
	(84, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'train', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'train'),
	(85, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'truck', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'truck'),
	(86, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'boat', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '62', '136', 'boat'),
	(87, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'person', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '63', '136', 'person'),
	(88, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'fire hydrant', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '63', '136', 'fire hydrant'),
	(89, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'stop sign', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '63', '136', 'stop sign'),
	(90, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'parking meter', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '63', '136', 'parking meter');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(91, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bird', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '64', '136', 'bird'),
	(92, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'cat', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '65', '136', 'cat'),
	(93, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'dog', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '66', '136', 'dog'),
	(94, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'horse', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '67', '136', 'horse'),
	(95, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'sheep', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '68', '136', 'sheep'),
	(96, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'cow', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '69', '136', 'cow'),
	(97, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'elephant', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '70', '136', 'elephant'),
	(98, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bear', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '71', '136', 'bear'),
	(99, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'zebra', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '72', '136', 'zebra'),
	(100, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'backpack', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '73', '136', 'backpack');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(101, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'umbrella', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '73', '136', 'umbrella'),
	(102, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'handbag', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '73', '136', 'handbag'),
	(103, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'tie', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '73', '136', 'tie'),
	(104, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'suitcase', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '73', '136', 'suitcase'),
	(105, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'frisbee', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '74', '136', 'frisbee'),
	(106, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'skis', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '74', '136', 'skis'),
	(107, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'snowboard', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '74', '136', 'snowboard'),
	(108, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'sports ball', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '74', '136', 'sports ball'),
	(109, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'kite', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '74', '136', 'kite'),
	(110, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'baseball bat', 'D', 'I', 'tiny_yolo', 'mscoco', '14', '74', '136', 'baseball bat');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(111, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'baseball glove', 'D', 'I', 'tiny_yolo', 'mscoco', '14', '74', '136', 'baseball glove'),
	(112, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'skateboard', 'D', 'I', 'tiny_yolo', 'mscoco', '14', '74', '136', 'skateboard'),
	(113, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'surfboard', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '74', '136', 'surfboard'),
	(114, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'tennis racket', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '74', '136', 'tennis racket'),
	(115, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bottle', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '78', '136', 'bottle'),
	(116, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'wine glass', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '78', '136', 'wine glass'),
	(117, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'cup', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '78', '136', 'cup'),
	(118, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'fork', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '78', '136', 'fork'),
	(119, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'knife', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '78', '136', 'knife'),
	(120, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'spoon', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '78', '136', 'spoon');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(121, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bowl', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '78', '136', 'bowl'),
	(122, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'banana', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'banana'),
	(123, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'sandwich', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'sandwich'),
	(124, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'orange', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'orange'),
	(125, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'broccoli', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'broccoli'),
	(126, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'carrot', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'carrot'),
	(127, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'hot dog', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'hot dog'),
	(128, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'pizza', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'pizza'),
	(129, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'donut', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'donut'),
	(130, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'cake', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'cake');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(131, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'chair', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '80', '136', 'chair'),
	(132, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'sofa', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '80', '136', 'sofa'),
	(133, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'pottedplant', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '81', '136', 'pottedplant'),
	(134, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bed', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '80', '136', 'bed'),
	(135, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'diningtable', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '80', '136', 'diningtable'),
	(136, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'toilet', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '80', '136', 'toilet'),
	(137, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'tvmonitor', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'tvmonitor'),
	(138, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'laptop', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'laptop'),
	(139, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'mouse', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'mouse'),
	(140, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'remote', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'remote');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(141, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'keyboard', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'keyboard'),
	(142, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'cell phone', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'cell phone'),
	(143, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'microwave', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'microwave'),
	(144, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'oven', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'oven'),
	(145, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'toaster', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'toaster'),
	(146, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'sink', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'sink'),
	(147, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'refrigerator', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '82', '136', 'refrigerator'),
	(148, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'book', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '84', '136', 'book'),
	(149, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'clock', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '84', '136', 'clock'),
	(150, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'vase', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '84', '136', 'vase');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(151, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'scissors', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '84', '136', 'scissors'),
	(152, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'teddy bear', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '84', '136', 'teddy bear'),
	(153, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'hair drier', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '84', '136', 'hair drier'),
	(154, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'toothbrush', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '84', '136', 'toothbrush'),
	(155, '/var/appdata/models/segmentation/deeplab/', 'aeroplane', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '86', '161', 'aeroplane'),
	(156, '/var/appdata/models/segmentation/deeplab/', 'bicycle', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '86', '161', 'bicycle'),
	(157, '/var/appdata/models/segmentation/deeplab/', 'bird', 'S', 'I', 'DEEP-LAB', 'pascal', '16', '87', '161', 'bird'),
	(158, '/var/appdata/models/segmentation/deeplab/', 'boat', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '86', '161', 'boat'),
	(159, '/var/appdata/models/segmentation/deeplab/', 'bottle', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '88', '161', 'bottle'),
	(160, '/var/appdata/models/segmentation/deeplab/', 'bus', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '86', '161', 'bus');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(161, '/var/appdata/models/segmentation/deeplab/', 'car', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '86', '161', 'car'),
	(162, '/var/appdata/models/segmentation/deeplab/', 'cat', 'S', 'I', 'DEEP-LAB', 'pascal', '16', '89', '161', 'cat'),
	(163, '/var/appdata/models/segmentation/deeplab/', 'chair', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '90', '161', 'chair'),
	(164, '/var/appdata/models/segmentation/deeplab/', 'cow', 'S', 'I', 'DEEP-LAB', 'pascal', '16', '91', '161', 'cow'),
	(165, '/var/appdata/models/segmentation/deeplab/', 'diningtable', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '90', '161', 'diningtable'),
	(166, '/var/appdata/models/segmentation/deeplab/', 'dog', 'S', 'I', 'DEEP-LAB', 'pascal', '16', '92', '161', 'dog'),
	(167, '/var/appdata/models/segmentation/deeplab/', 'horse', 'S', 'I', 'DEEP-LAB', 'pascal', '16', '93', '161', 'horse'),
	(168, '/var/appdata/models/segmentation/deeplab/', 'motorbike', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '86', '161', 'motorbike'),
	(169, '/var/appdata/models/segmentation/deeplab/', 'person', 'S', 'I', 'DEEP-LAB', 'pascal', '15', '15', '161', 'person'),
	(170, '/var/appdata/models/segmentation/deeplab/', 'pottedplant', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '90', '161', 'pottedplant');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(171, '/var/appdata/models/segmentation/deeplab/', 'sheep', 'S', 'I', 'DEEP-LAB', 'pascal', '16', '95', '161', 'sheep'),
	(172, '/var/appdata/models/segmentation/deeplab/', 'sofa', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '90', '161', 'sofa'),
	(173, '/var/appdata/models/segmentation/deeplab/', 'train', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '86', '161', 'train'),
	(174, '/var/appdata/models/segmentation/deeplab/', 'tvmonitor', 'S', 'I', 'DEEP-LAB', 'pascal', '17', '96', '161', 'tvmonitor'),
	(175, '/var/appdata/models/classification/efficientnet/', 'English_setter', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'English_setter'),
	(176, '/var/appdata/models/classification/efficientnet/', 'Siberian_husky', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Siberian_husky'),
	(177, '/var/appdata/models/classification/efficientnet/', 'Australian_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Australian_terrier'),
	(178, '/var/appdata/models/classification/efficientnet/', 'English_springer', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'English_springer'),
	(179, '/var/appdata/models/classification/efficientnet/', 'lesser_panda', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'lesser_panda'),
	(180, '/var/appdata/models/classification/efficientnet/', 'Egyptian_cat', 'C', 'I', 'efficientnet', 'imagenet', '2', '24', '172', 'Egyptian_cat');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(181, '/var/appdata/models/classification/efficientnet/', 'ibex', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'ibex'),
	(182, '/var/appdata/models/classification/efficientnet/', 'Persian_cat', 'C', 'I', 'efficientnet', 'imagenet', '2', '24', '172', 'Persian_cat'),
	(183, '/var/appdata/models/classification/efficientnet/', 'cougar', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'cougar'),
	(184, '/var/appdata/models/classification/efficientnet/', 'gazelle', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'gazelle'),
	(185, '/var/appdata/models/classification/efficientnet/', 'porcupine', 'C', 'I', 'efficientnet', 'imagenet', '2', '25', '172', 'porcupine'),
	(186, '/var/appdata/models/classification/efficientnet/', 'sea_lion', 'C', 'I', 'efficientnet', 'imagenet', '2', '26', '172', 'sea_lion'),
	(187, '/var/appdata/models/classification/efficientnet/', 'grey_whale', 'C', 'I', 'efficientnet', 'imagenet', '2', '26', '172', 'grey_whale'),
	(188, '/var/appdata/models/classification/efficientnet/', 'malamute', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'malamute'),
	(189, '/var/appdata/models/classification/efficientnet/', 'badger', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'badger'),
	(190, '/var/appdata/models/classification/efficientnet/', 'Great_Dane', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Great_Dane');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(191, '/var/appdata/models/classification/efficientnet/', 'Walker_hound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Walker_hound'),
	(192, '/var/appdata/models/classification/efficientnet/', 'Welsh_springer_spaniel', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Welsh_springer_spaniel'),
	(193, '/var/appdata/models/classification/efficientnet/', 'whippet', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'whippet'),
	(194, '/var/appdata/models/classification/efficientnet/', 'Scottish_deerhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Scottish_deerhound'),
	(195, '/var/appdata/models/classification/efficientnet/', 'killer_whale', 'C', 'I', 'efficientnet', 'imagenet', '2', '26', '172', 'killer_whale'),
	(196, '/var/appdata/models/classification/efficientnet/', 'mink', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'mink'),
	(197, '/var/appdata/models/classification/efficientnet/', 'African_elephant', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'African_elephant'),
	(198, '/var/appdata/models/classification/efficientnet/', 'Weimaraner', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Weimaraner'),
	(199, '/var/appdata/models/classification/efficientnet/', 'soft-coated_wheaten_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'soft-coated_wheaten_terrier'),
	(200, '/var/appdata/models/classification/efficientnet/', 'Dandie_Dinmont', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Dandie_Dinmont');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(201, '/var/appdata/models/classification/efficientnet/', 'red_wolf', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'red_wolf'),
	(202, '/var/appdata/models/classification/efficientnet/', 'Old_English_sheepdog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Old_English_sheepdog'),
	(203, '/var/appdata/models/classification/efficientnet/', 'jaguar', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'jaguar'),
	(204, '/var/appdata/models/classification/efficientnet/', 'otterhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'otterhound'),
	(205, '/var/appdata/models/classification/efficientnet/', 'bloodhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'bloodhound'),
	(206, '/var/appdata/models/classification/efficientnet/', 'Airedale', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Airedale'),
	(207, '/var/appdata/models/classification/efficientnet/', 'hyena', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'hyena'),
	(208, '/var/appdata/models/classification/efficientnet/', 'meerkat', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'meerkat'),
	(209, '/var/appdata/models/classification/efficientnet/', 'giant_schnauzer', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'giant_schnauzer'),
	(210, '/var/appdata/models/classification/efficientnet/', 'titi', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'titi');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(211, '/var/appdata/models/classification/efficientnet/', 'three-toed_sloth', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'three-toed_sloth'),
	(212, '/var/appdata/models/classification/efficientnet/', 'sorrel', 'C', 'I', 'efficientnet', 'imagenet', '4', '28', '172', 'sorrel'),
	(213, '/var/appdata/models/classification/efficientnet/', 'black-footed_ferret', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'black-footed_ferret'),
	(214, '/var/appdata/models/classification/efficientnet/', 'dalmatian', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'dalmatian'),
	(215, '/var/appdata/models/classification/efficientnet/', 'black-and-tan_coonhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'black-and-tan_coonhound'),
	(216, '/var/appdata/models/classification/efficientnet/', 'papillon', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'papillon'),
	(217, '/var/appdata/models/classification/efficientnet/', 'skunk', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'skunk'),
	(218, '/var/appdata/models/classification/efficientnet/', 'Staffordshire_bullterrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Staffordshire_bullterrier'),
	(219, '/var/appdata/models/classification/efficientnet/', 'Mexican_hairless', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Mexican_hairless'),
	(220, '/var/appdata/models/classification/efficientnet/', 'Bouvier_des_Flandres', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Bouvier_des_Flandres');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(221, '/var/appdata/models/classification/efficientnet/', 'weasel', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'weasel'),
	(222, '/var/appdata/models/classification/efficientnet/', 'miniature_poodle', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'miniature_poodle'),
	(223, '/var/appdata/models/classification/efficientnet/', 'Cardigan', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Cardigan'),
	(224, '/var/appdata/models/classification/efficientnet/', 'malinois', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'malinois'),
	(225, '/var/appdata/models/classification/efficientnet/', 'bighorn', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'bighorn'),
	(226, '/var/appdata/models/classification/efficientnet/', 'fox_squirrel', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'fox_squirrel'),
	(227, '/var/appdata/models/classification/efficientnet/', 'colobus', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'colobus'),
	(228, '/var/appdata/models/classification/efficientnet/', 'tiger_cat', 'C', 'I', 'efficientnet', 'imagenet', '2', '24', '172', 'tiger_cat'),
	(229, '/var/appdata/models/classification/efficientnet/', 'Lhasa', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Lhasa'),
	(230, '/var/appdata/models/classification/efficientnet/', 'impala', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'impala');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(231, '/var/appdata/models/classification/efficientnet/', 'coyote', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'coyote'),
	(232, '/var/appdata/models/classification/efficientnet/', 'Yorkshire_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Yorkshire_terrier'),
	(233, '/var/appdata/models/classification/efficientnet/', 'Newfoundland', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Newfoundland'),
	(234, '/var/appdata/models/classification/efficientnet/', 'brown_bear', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'brown_bear'),
	(235, '/var/appdata/models/classification/efficientnet/', 'red_fox', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'red_fox'),
	(236, '/var/appdata/models/classification/efficientnet/', 'Norwegian_elkhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Norwegian_elkhound'),
	(237, '/var/appdata/models/classification/efficientnet/', 'Rottweiler', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Rottweiler'),
	(238, '/var/appdata/models/classification/efficientnet/', 'hartebeest', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'hartebeest'),
	(239, '/var/appdata/models/classification/efficientnet/', 'Saluki', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'Saluki'),
	(240, '/var/appdata/models/classification/efficientnet/', 'grey_fox', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'grey_fox');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(241, '/var/appdata/models/classification/efficientnet/', 'schipperke', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'schipperke'),
	(242, '/var/appdata/models/classification/efficientnet/', 'Pekinese', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Pekinese'),
	(243, '/var/appdata/models/classification/efficientnet/', 'Brabancon_griffon', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Brabancon_griffon'),
	(244, '/var/appdata/models/classification/efficientnet/', 'West_Highland_white_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'West_Highland_white_terrier'),
	(245, '/var/appdata/models/classification/efficientnet/', 'Sealyham_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Sealyham_terrier'),
	(246, '/var/appdata/models/classification/efficientnet/', 'guenon', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'guenon'),
	(247, '/var/appdata/models/classification/efficientnet/', 'mongoose', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'mongoose'),
	(248, '/var/appdata/models/classification/efficientnet/', 'indri', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'indri'),
	(249, '/var/appdata/models/classification/efficientnet/', 'tiger', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'tiger'),
	(250, '/var/appdata/models/classification/efficientnet/', 'Irish_wolfhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Irish_wolfhound');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(251, '/var/appdata/models/classification/efficientnet/', 'wild_boar', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'wild_boar'),
	(252, '/var/appdata/models/classification/efficientnet/', 'EntleBucher', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'EntleBucher'),
	(253, '/var/appdata/models/classification/efficientnet/', 'zebra', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'zebra'),
	(254, '/var/appdata/models/classification/efficientnet/', 'ram', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'ram'),
	(255, '/var/appdata/models/classification/efficientnet/', 'French_bulldog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'French_bulldog'),
	(256, '/var/appdata/models/classification/efficientnet/', 'orangutan', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'orangutan'),
	(257, '/var/appdata/models/classification/efficientnet/', 'basenji', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'basenji'),
	(258, '/var/appdata/models/classification/efficientnet/', 'leopard', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'leopard'),
	(259, '/var/appdata/models/classification/efficientnet/', 'Bernese_mountain_dog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Bernese_mountain_dog'),
	(260, '/var/appdata/models/classification/efficientnet/', 'Maltese_dog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Maltese_dog');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(261, '/var/appdata/models/classification/efficientnet/', 'Norfolk_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Norfolk_terrier'),
	(262, '/var/appdata/models/classification/efficientnet/', 'toy_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'toy_terrier'),
	(263, '/var/appdata/models/classification/efficientnet/', 'vizsla', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'vizsla'),
	(264, '/var/appdata/models/classification/efficientnet/', 'cairn', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'cairn'),
	(265, '/var/appdata/models/classification/efficientnet/', 'squirrel_monkey', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'squirrel_monkey'),
	(266, '/var/appdata/models/classification/efficientnet/', 'groenendael', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'groenendael'),
	(267, '/var/appdata/models/classification/efficientnet/', 'clumber', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'clumber'),
	(268, '/var/appdata/models/classification/efficientnet/', 'Siamese_cat', 'C', 'I', 'efficientnet', 'imagenet', '2', '24', '172', 'Siamese_cat'),
	(269, '/var/appdata/models/classification/efficientnet/', 'chimpanzee', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'chimpanzee'),
	(270, '/var/appdata/models/classification/efficientnet/', 'komondor', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'komondor');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(271, '/var/appdata/models/classification/efficientnet/', 'Afghan_hound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Afghan_hound'),
	(272, '/var/appdata/models/classification/efficientnet/', 'Japanese_spaniel', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Japanese_spaniel'),
	(273, '/var/appdata/models/classification/efficientnet/', 'proboscis_monkey', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'proboscis_monkey'),
	(274, '/var/appdata/models/classification/efficientnet/', 'guinea_pig', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'guinea_pig'),
	(275, '/var/appdata/models/classification/efficientnet/', 'white_wolf', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'white_wolf'),
	(276, '/var/appdata/models/classification/efficientnet/', 'ice_bear', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'ice_bear'),
	(277, '/var/appdata/models/classification/efficientnet/', 'gorilla', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'gorilla'),
	(278, '/var/appdata/models/classification/efficientnet/', 'borzoi', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'borzoi'),
	(279, '/var/appdata/models/classification/efficientnet/', 'toy_poodle', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'toy_poodle'),
	(280, '/var/appdata/models/classification/efficientnet/', 'Kerry_blue_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Kerry_blue_terrier');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(281, '/var/appdata/models/classification/efficientnet/', 'ox', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'ox'),
	(282, '/var/appdata/models/classification/efficientnet/', 'Scotch_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Scotch_terrier'),
	(283, '/var/appdata/models/classification/efficientnet/', 'Tibetan_mastiff', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Tibetan_mastiff'),
	(284, '/var/appdata/models/classification/efficientnet/', 'spider_monkey', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'spider_monkey'),
	(285, '/var/appdata/models/classification/efficientnet/', 'Doberman', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Doberman'),
	(286, '/var/appdata/models/classification/efficientnet/', 'Boston_bull', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Boston_bull'),
	(287, '/var/appdata/models/classification/efficientnet/', 'Greater_Swiss_Mountain_dog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Greater_Swiss_Mountain_dog'),
	(288, '/var/appdata/models/classification/efficientnet/', 'Appenzeller', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Appenzeller'),
	(289, '/var/appdata/models/classification/efficientnet/', 'Shih-Tzu', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Shih-Tzu'),
	(290, '/var/appdata/models/classification/efficientnet/', 'Irish_water_spaniel', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Irish_water_spaniel');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(291, '/var/appdata/models/classification/efficientnet/', 'Pomeranian', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Pomeranian'),
	(292, '/var/appdata/models/classification/efficientnet/', 'Bedlington_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Bedlington_terrier'),
	(293, '/var/appdata/models/classification/efficientnet/', 'warthog', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'warthog'),
	(294, '/var/appdata/models/classification/efficientnet/', 'Arabian_camel', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'Arabian_camel'),
	(295, '/var/appdata/models/classification/efficientnet/', 'siamang', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'siamang'),
	(296, '/var/appdata/models/classification/efficientnet/', 'miniature_schnauzer', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'miniature_schnauzer'),
	(297, '/var/appdata/models/classification/efficientnet/', 'collie', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'collie'),
	(298, '/var/appdata/models/classification/efficientnet/', 'golden_retriever', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'golden_retriever'),
	(299, '/var/appdata/models/classification/efficientnet/', 'Irish_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Irish_terrier'),
	(300, '/var/appdata/models/classification/efficientnet/', 'affenpinscher', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'affenpinscher');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(301, '/var/appdata/models/classification/efficientnet/', 'Border_collie', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Border_collie'),
	(302, '/var/appdata/models/classification/efficientnet/', 'hare', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'hare'),
	(303, '/var/appdata/models/classification/efficientnet/', 'boxer', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'boxer'),
	(304, '/var/appdata/models/classification/efficientnet/', 'silky_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'silky_terrier'),
	(305, '/var/appdata/models/classification/efficientnet/', 'beagle', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'beagle'),
	(306, '/var/appdata/models/classification/efficientnet/', 'Leonberg', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Leonberg'),
	(307, '/var/appdata/models/classification/efficientnet/', 'German_short-haired_pointer', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'German_short-haired_pointer'),
	(308, '/var/appdata/models/classification/efficientnet/', 'patas', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'patas'),
	(309, '/var/appdata/models/classification/efficientnet/', 'dhole', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'dhole'),
	(310, '/var/appdata/models/classification/efficientnet/', 'baboon', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'baboon');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(311, '/var/appdata/models/classification/efficientnet/', 'macaque', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'macaque'),
	(312, '/var/appdata/models/classification/efficientnet/', 'Chesapeake_Bay_retriever', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Chesapeake_Bay_retriever'),
	(313, '/var/appdata/models/classification/efficientnet/', 'bull_mastiff', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'bull_mastiff'),
	(314, '/var/appdata/models/classification/efficientnet/', 'kuvasz', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'kuvasz'),
	(315, '/var/appdata/models/classification/efficientnet/', 'capuchin', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'capuchin'),
	(316, '/var/appdata/models/classification/efficientnet/', 'pug', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'pug'),
	(317, '/var/appdata/models/classification/efficientnet/', 'curly-coated_retriever', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'curly-coated_retriever'),
	(318, '/var/appdata/models/classification/efficientnet/', 'Norwich_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Norwich_terrier'),
	(319, '/var/appdata/models/classification/efficientnet/', 'flat-coated_retriever', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'flat-coated_retriever'),
	(320, '/var/appdata/models/classification/efficientnet/', 'hog', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'hog');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(321, '/var/appdata/models/classification/efficientnet/', 'keeshond', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'keeshond'),
	(322, '/var/appdata/models/classification/efficientnet/', 'Eskimo_dog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Eskimo_dog'),
	(323, '/var/appdata/models/classification/efficientnet/', 'Brittany_spaniel', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Brittany_spaniel'),
	(324, '/var/appdata/models/classification/efficientnet/', 'standard_poodle', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'standard_poodle'),
	(325, '/var/appdata/models/classification/efficientnet/', 'Lakeland_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Lakeland_terrier'),
	(326, '/var/appdata/models/classification/efficientnet/', 'snow_leopard', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'snow_leopard'),
	(327, '/var/appdata/models/classification/efficientnet/', 'Gordon_setter', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Gordon_setter'),
	(328, '/var/appdata/models/classification/efficientnet/', 'dingo', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'dingo'),
	(329, '/var/appdata/models/classification/efficientnet/', 'standard_schnauzer', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'standard_schnauzer'),
	(330, '/var/appdata/models/classification/efficientnet/', 'hamster', 'C', 'I', 'efficientnet', 'imagenet', '2', '25', '172', 'hamster');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(331, '/var/appdata/models/classification/efficientnet/', 'Tibetan_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Tibetan_terrier'),
	(332, '/var/appdata/models/classification/efficientnet/', 'Arctic_fox', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'Arctic_fox'),
	(333, '/var/appdata/models/classification/efficientnet/', 'wire-haired_fox_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'wire-haired_fox_terrier'),
	(334, '/var/appdata/models/classification/efficientnet/', 'basset', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'basset'),
	(335, '/var/appdata/models/classification/efficientnet/', 'water_buffalo', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'water_buffalo'),
	(336, '/var/appdata/models/classification/efficientnet/', 'American_black_bear', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'American_black_bear'),
	(337, '/var/appdata/models/classification/efficientnet/', 'Angora', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'Angora'),
	(338, '/var/appdata/models/classification/efficientnet/', 'bison', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'bison'),
	(339, '/var/appdata/models/classification/efficientnet/', 'howler_monkey', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'howler_monkey'),
	(340, '/var/appdata/models/classification/efficientnet/', 'hippopotamus', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'hippopotamus');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(341, '/var/appdata/models/classification/efficientnet/', 'chow', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'chow'),
	(342, '/var/appdata/models/classification/efficientnet/', 'giant_panda', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'giant_panda'),
	(343, '/var/appdata/models/classification/efficientnet/', 'American_Staffordshire_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'American_Staffordshire_terrier'),
	(344, '/var/appdata/models/classification/efficientnet/', 'Shetland_sheepdog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Shetland_sheepdog'),
	(345, '/var/appdata/models/classification/efficientnet/', 'Great_Pyrenees', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Great_Pyrenees'),
	(346, '/var/appdata/models/classification/efficientnet/', 'Chihuahua', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Chihuahua'),
	(347, '/var/appdata/models/classification/efficientnet/', 'tabby', 'C', 'I', 'efficientnet', 'imagenet', '2', '24', '172', 'tabby'),
	(348, '/var/appdata/models/classification/efficientnet/', 'marmoset', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'marmoset'),
	(349, '/var/appdata/models/classification/efficientnet/', 'Labrador_retriever', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Labrador_retriever'),
	(350, '/var/appdata/models/classification/efficientnet/', 'Saint_Bernard', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Saint_Bernard');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(351, '/var/appdata/models/classification/efficientnet/', 'armadillo', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'armadillo'),
	(352, '/var/appdata/models/classification/efficientnet/', 'Samoyed', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Samoyed'),
	(353, '/var/appdata/models/classification/efficientnet/', 'bluetick', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'bluetick'),
	(354, '/var/appdata/models/classification/efficientnet/', 'redbone', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'redbone'),
	(355, '/var/appdata/models/classification/efficientnet/', 'polecat', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'polecat'),
	(356, '/var/appdata/models/classification/efficientnet/', 'marmot', 'C', 'I', 'efficientnet', 'imagenet', '2', '25', '172', 'marmot'),
	(357, '/var/appdata/models/classification/efficientnet/', 'kelpie', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'kelpie'),
	(358, '/var/appdata/models/classification/efficientnet/', 'gibbon', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'gibbon'),
	(359, '/var/appdata/models/classification/efficientnet/', 'llama', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'llama'),
	(360, '/var/appdata/models/classification/efficientnet/', 'miniature_pinscher', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'miniature_pinscher');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(361, '/var/appdata/models/classification/efficientnet/', 'wood_rabbit', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'wood_rabbit'),
	(362, '/var/appdata/models/classification/efficientnet/', 'Italian_greyhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Italian_greyhound'),
	(363, '/var/appdata/models/classification/efficientnet/', 'lion', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'lion'),
	(364, '/var/appdata/models/classification/efficientnet/', 'cocker_spaniel', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'cocker_spaniel'),
	(365, '/var/appdata/models/classification/efficientnet/', 'Irish_setter', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Irish_setter'),
	(366, '/var/appdata/models/classification/efficientnet/', 'dugong', 'C', 'I', 'efficientnet', 'imagenet', '2', '26', '172', 'dugong'),
	(367, '/var/appdata/models/classification/efficientnet/', 'Indian_elephant', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'Indian_elephant'),
	(368, '/var/appdata/models/classification/efficientnet/', 'beaver', 'C', 'I', 'efficientnet', 'imagenet', '2', '25', '172', 'beaver'),
	(369, '/var/appdata/models/classification/efficientnet/', 'Sussex_spaniel', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Sussex_spaniel'),
	(370, '/var/appdata/models/classification/efficientnet/', 'Pembroke', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Pembroke');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(371, '/var/appdata/models/classification/efficientnet/', 'Blenheim_spaniel', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Blenheim_spaniel'),
	(372, '/var/appdata/models/classification/efficientnet/', 'Madagascar_cat', 'C', 'I', 'efficientnet', 'imagenet', '2', '24', '172', 'Madagascar_cat'),
	(373, '/var/appdata/models/classification/efficientnet/', 'Rhodesian_ridgeback', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Rhodesian_ridgeback'),
	(374, '/var/appdata/models/classification/efficientnet/', 'lynx', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'lynx'),
	(375, '/var/appdata/models/classification/efficientnet/', 'African_hunting_dog', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'African_hunting_dog'),
	(376, '/var/appdata/models/classification/efficientnet/', 'langur', 'C', 'I', 'efficientnet', 'imagenet', '2', '27', '172', 'langur'),
	(377, '/var/appdata/models/classification/efficientnet/', 'Ibizan_hound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Ibizan_hound'),
	(378, '/var/appdata/models/classification/efficientnet/', 'timber_wolf', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'timber_wolf'),
	(379, '/var/appdata/models/classification/efficientnet/', 'cheetah', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'cheetah'),
	(380, '/var/appdata/models/classification/efficientnet/', 'English_foxhound', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'English_foxhound');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(381, '/var/appdata/models/classification/efficientnet/', 'briard', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'briard'),
	(382, '/var/appdata/models/classification/efficientnet/', 'sloth_bear', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'sloth_bear'),
	(383, '/var/appdata/models/classification/efficientnet/', 'Border_terrier', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'Border_terrier'),
	(384, '/var/appdata/models/classification/efficientnet/', 'German_shepherd', 'C', 'I', 'efficientnet', 'imagenet', '2', '22', '172', 'German_shepherd'),
	(385, '/var/appdata/models/classification/efficientnet/', 'otter', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'otter'),
	(386, '/var/appdata/models/classification/efficientnet/', 'koala', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'koala'),
	(387, '/var/appdata/models/classification/efficientnet/', 'tusker', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'tusker'),
	(388, '/var/appdata/models/classification/efficientnet/', 'echidna', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'echidna'),
	(389, '/var/appdata/models/classification/efficientnet/', 'wallaby', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'wallaby'),
	(390, '/var/appdata/models/classification/efficientnet/', 'platypus', 'C', 'I', 'efficientnet', 'imagenet', '2', '26', '172', 'platypus');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(391, '/var/appdata/models/classification/efficientnet/', 'wombat', 'C', 'I', 'efficientnet', 'imagenet', '2', '23', '172', 'wombat'),
	(392, '/var/appdata/models/classification/efficientnet/', 'revolver', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'revolver'),
	(393, '/var/appdata/models/classification/efficientnet/', 'umbrella', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'umbrella'),
	(394, '/var/appdata/models/classification/efficientnet/', 'schooner', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'schooner'),
	(395, '/var/appdata/models/classification/efficientnet/', 'soccer_ball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'soccer_ball'),
	(396, '/var/appdata/models/classification/efficientnet/', 'accordion', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'accordion'),
	(397, '/var/appdata/models/classification/efficientnet/', 'ant', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'ant'),
	(398, '/var/appdata/models/classification/efficientnet/', 'starfish', 'C', 'I', 'efficientnet', 'imagenet', '2', '35', '172', 'starfish'),
	(399, '/var/appdata/models/classification/efficientnet/', 'chambered_nautilus', 'C', 'I', 'efficientnet', 'imagenet', '2', '36', '172', 'chambered_nautilus'),
	(400, '/var/appdata/models/classification/efficientnet/', 'grand_piano', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'grand_piano');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(401, '/var/appdata/models/classification/efficientnet/', 'laptop', 'C', 'I', 'efficientnet', 'imagenet', '3', '37', '172', 'laptop'),
	(402, '/var/appdata/models/classification/efficientnet/', 'strawberry', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'strawberry'),
	(403, '/var/appdata/models/classification/efficientnet/', 'airliner', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'airliner'),
	(404, '/var/appdata/models/classification/efficientnet/', 'warplane', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'warplane'),
	(405, '/var/appdata/models/classification/efficientnet/', 'airship', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'airship'),
	(406, '/var/appdata/models/classification/efficientnet/', 'balloon', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'balloon'),
	(407, '/var/appdata/models/classification/efficientnet/', 'fireboat', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'fireboat'),
	(408, '/var/appdata/models/classification/efficientnet/', 'gondola', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'gondola'),
	(409, '/var/appdata/models/classification/efficientnet/', 'speedboat', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'speedboat'),
	(410, '/var/appdata/models/classification/efficientnet/', 'lifeboat', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'lifeboat');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(411, '/var/appdata/models/classification/efficientnet/', 'canoe', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'canoe'),
	(412, '/var/appdata/models/classification/efficientnet/', 'yawl', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'yawl'),
	(413, '/var/appdata/models/classification/efficientnet/', 'catamaran', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'catamaran'),
	(414, '/var/appdata/models/classification/efficientnet/', 'trimaran', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'trimaran'),
	(415, '/var/appdata/models/classification/efficientnet/', 'container_ship', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'container_ship'),
	(416, '/var/appdata/models/classification/efficientnet/', 'liner', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'liner'),
	(417, '/var/appdata/models/classification/efficientnet/', 'pirate', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'pirate'),
	(418, '/var/appdata/models/classification/efficientnet/', 'aircraft_carrier', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'aircraft_carrier'),
	(419, '/var/appdata/models/classification/efficientnet/', 'submarine', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'submarine'),
	(420, '/var/appdata/models/classification/efficientnet/', 'wreck', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'wreck');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(421, '/var/appdata/models/classification/efficientnet/', 'half_track', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'half_track'),
	(422, '/var/appdata/models/classification/efficientnet/', 'tank', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'tank'),
	(423, '/var/appdata/models/classification/efficientnet/', 'missile', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'missile'),
	(424, '/var/appdata/models/classification/efficientnet/', 'freight_car', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'freight_car'),
	(425, '/var/appdata/models/classification/efficientnet/', 'passenger_car', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'passenger_car'),
	(426, '/var/appdata/models/classification/efficientnet/', 'barrow', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'barrow'),
	(427, '/var/appdata/models/classification/efficientnet/', 'shopping_cart', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'shopping_cart'),
	(428, '/var/appdata/models/classification/efficientnet/', 'motor_scooter', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'motor_scooter'),
	(429, '/var/appdata/models/classification/efficientnet/', 'forklift', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'forklift'),
	(430, '/var/appdata/models/classification/efficientnet/', 'electric_locomotive', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'electric_locomotive');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(431, '/var/appdata/models/classification/efficientnet/', 'steam_locomotive', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'steam_locomotive'),
	(432, '/var/appdata/models/classification/efficientnet/', 'amphibian', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'amphibian'),
	(433, '/var/appdata/models/classification/efficientnet/', 'ambulance', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'ambulance'),
	(434, '/var/appdata/models/classification/efficientnet/', 'beach_wagon', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'beach_wagon'),
	(435, '/var/appdata/models/classification/efficientnet/', 'cab', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'cab'),
	(436, '/var/appdata/models/classification/efficientnet/', 'convertible', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'convertible'),
	(437, '/var/appdata/models/classification/efficientnet/', 'jeep', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'jeep'),
	(438, '/var/appdata/models/classification/efficientnet/', 'limousine', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'limousine'),
	(439, '/var/appdata/models/classification/efficientnet/', 'minivan', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'minivan'),
	(440, '/var/appdata/models/classification/efficientnet/', 'Model_T', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'Model_T');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(441, '/var/appdata/models/classification/efficientnet/', 'racer', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'racer'),
	(442, '/var/appdata/models/classification/efficientnet/', 'sports_car', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'sports_car'),
	(443, '/var/appdata/models/classification/efficientnet/', 'go-kart', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'go-kart'),
	(444, '/var/appdata/models/classification/efficientnet/', 'golfcart', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'golfcart'),
	(445, '/var/appdata/models/classification/efficientnet/', 'moped', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'moped'),
	(446, '/var/appdata/models/classification/efficientnet/', 'snowplow', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'snowplow'),
	(447, '/var/appdata/models/classification/efficientnet/', 'fire_engine', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'fire_engine'),
	(448, '/var/appdata/models/classification/efficientnet/', 'garbage_truck', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'garbage_truck'),
	(449, '/var/appdata/models/classification/efficientnet/', 'pickup', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'pickup'),
	(450, '/var/appdata/models/classification/efficientnet/', 'tow_truck', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'tow_truck');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(451, '/var/appdata/models/classification/efficientnet/', 'trailer_truck', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'trailer_truck'),
	(452, '/var/appdata/models/classification/efficientnet/', 'moving_van', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'moving_van'),
	(453, '/var/appdata/models/classification/efficientnet/', 'police_van-kart', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'police_van-kart'),
	(454, '/var/appdata/models/classification/efficientnet/', 'recreational_vehicle', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'recreational_vehicle'),
	(455, '/var/appdata/models/classification/efficientnet/', 'streetcar', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'streetcar'),
	(456, '/var/appdata/models/classification/efficientnet/', 'snowmobile', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'snowmobile'),
	(457, '/var/appdata/models/classification/efficientnet/', 'tractor', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'tractor'),
	(458, '/var/appdata/models/classification/efficientnet/', 'mobile_home', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'mobile_home'),
	(459, '/var/appdata/models/classification/efficientnet/', 'tricycle', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'tricycle'),
	(460, '/var/appdata/models/classification/efficientnet/', 'unicycle', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'unicycle');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(461, '/var/appdata/models/classification/efficientnet/', 'horse_cart', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'horse_cart'),
	(462, '/var/appdata/models/classification/efficientnet/', 'jinrikisha', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'jinrikisha'),
	(463, '/var/appdata/models/classification/efficientnet/', 'oxcart-kart', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'oxcart-kart'),
	(464, '/var/appdata/models/classification/efficientnet/', 'bassinet', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'bassinet'),
	(465, '/var/appdata/models/classification/efficientnet/', 'cradle', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'cradle'),
	(466, '/var/appdata/models/classification/efficientnet/', 'crib', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'crib'),
	(467, '/var/appdata/models/classification/efficientnet/', 'four-poster', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'four-poster'),
	(468, '/var/appdata/models/classification/efficientnet/', 'bookcase', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'bookcase'),
	(469, '/var/appdata/models/classification/efficientnet/', 'china_cabinet', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'china_cabinet'),
	(470, '/var/appdata/models/classification/efficientnet/', 'medicine_chest', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'medicine_chest');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(471, '/var/appdata/models/classification/efficientnet/', 'chiffonier', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'chiffonier'),
	(472, '/var/appdata/models/classification/efficientnet/', 'table_lamp', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'table_lamp'),
	(473, '/var/appdata/models/classification/efficientnet/', 'file', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'file'),
	(474, '/var/appdata/models/classification/efficientnet/', 'bassinet', 'C', 'I', 'efficientnet', 'imagenet', '3', '41', '172', 'bassinet'),
	(475, '/var/appdata/models/classification/efficientnet/', 'barber_chair', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'barber_chair'),
	(476, '/var/appdata/models/classification/efficientnet/', 'throne', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'throne'),
	(477, '/var/appdata/models/classification/efficientnet/', 'folding_chair', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'folding_chair'),
	(478, '/var/appdata/models/classification/efficientnet/', 'rocking_chair', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'rocking_chair'),
	(479, '/var/appdata/models/classification/efficientnet/', 'studio_couch', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'studio_couch'),
	(480, '/var/appdata/models/classification/efficientnet/', 'desk', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'desk');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(481, '/var/appdata/models/classification/efficientnet/', 'pool_table', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pool_table'),
	(482, '/var/appdata/models/classification/efficientnet/', 'dining_table', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'dining_table'),
	(483, '/var/appdata/models/classification/efficientnet/', 'entertainment_center', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'entertainment_center'),
	(484, '/var/appdata/models/classification/efficientnet/', 'wardrobe', 'C', 'I', 'efficientnet', 'imagenet', '3', '40', '172', 'wardrobe'),
	(485, '/var/appdata/models/classification/efficientnet/', 'Granny_Smith', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'Granny_Smith'),
	(486, '/var/appdata/models/classification/efficientnet/', 'orange', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'orange'),
	(487, '/var/appdata/models/classification/efficientnet/', 'lemon', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'lemon'),
	(488, '/var/appdata/models/classification/efficientnet/', 'fig', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'fig'),
	(489, '/var/appdata/models/classification/efficientnet/', 'pineapple', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'pineapple'),
	(490, '/var/appdata/models/classification/efficientnet/', 'banana', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'banana');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(491, '/var/appdata/models/classification/efficientnet/', 'jackfruit', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'jackfruit'),
	(492, '/var/appdata/models/classification/efficientnet/', 'custard_apple', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'custard_apple'),
	(493, '/var/appdata/models/classification/efficientnet/', 'pomegranate', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'pomegranate'),
	(494, '/var/appdata/models/classification/efficientnet/', 'acorn', 'C', 'I', 'efficientnet', 'imagenet', '5', '38', '172', 'acorn'),
	(495, '/var/appdata/models/classification/efficientnet/', 'hip', 'C', 'I', 'efficientnet', 'imagenet', '7', '42', '172', 'hip'),
	(496, '/var/appdata/models/classification/efficientnet/', 'ear', 'C', 'I', 'efficientnet', 'imagenet', '7', '42', '172', 'ear'),
	(497, '/var/appdata/models/classification/efficientnet/', 'rapeseed', 'C', 'I', 'efficientnet', 'imagenet', '4', '43', '172', 'rapeseed'),
	(498, '/var/appdata/models/classification/efficientnet/', 'corn', 'C', 'I', 'efficientnet', 'imagenet', '4', '7', '172', 'corn'),
	(499, '/var/appdata/models/classification/efficientnet/', 'buckeye', 'C', 'I', 'efficientnet', 'imagenet', '4', '7', '172', 'buckeye'),
	(500, '/var/appdata/models/classification/efficientnet/', 'organ', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'organ');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(501, '/var/appdata/models/classification/efficientnet/', 'upright', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'upright'),
	(502, '/var/appdata/models/classification/efficientnet/', 'chime', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'chime'),
	(503, '/var/appdata/models/classification/efficientnet/', 'drum', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'drum'),
	(504, '/var/appdata/models/classification/efficientnet/', 'gong', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'gong'),
	(505, '/var/appdata/models/classification/efficientnet/', 'maraca', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'maraca'),
	(506, '/var/appdata/models/classification/efficientnet/', 'marimba', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'marimba'),
	(507, '/var/appdata/models/classification/efficientnet/', 'steel_drum', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'steel_drum'),
	(508, '/var/appdata/models/classification/efficientnet/', 'banjo', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'banjo'),
	(509, '/var/appdata/models/classification/efficientnet/', 'cello', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'cello'),
	(510, '/var/appdata/models/classification/efficientnet/', 'violin', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'violin');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(511, '/var/appdata/models/classification/efficientnet/', 'harp', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'harp'),
	(512, '/var/appdata/models/classification/efficientnet/', 'acoustic_guitar', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'acoustic_guitar'),
	(513, '/var/appdata/models/classification/efficientnet/', 'electric_guitar', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'electric_guitar'),
	(514, '/var/appdata/models/classification/efficientnet/', 'cornet', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'cornet'),
	(515, '/var/appdata/models/classification/efficientnet/', 'French_horn', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'French_horn'),
	(516, '/var/appdata/models/classification/efficientnet/', 'trombone', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'trombone'),
	(517, '/var/appdata/models/classification/efficientnet/', 'harmonica', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'harmonica'),
	(518, '/var/appdata/models/classification/efficientnet/', 'ocarina', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'ocarina'),
	(519, '/var/appdata/models/classification/efficientnet/', 'panpipe', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'panpipe'),
	(520, '/var/appdata/models/classification/efficientnet/', 'bassoon', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'bassoon');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(521, '/var/appdata/models/classification/efficientnet/', 'oboe', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'oboe'),
	(522, '/var/appdata/models/classification/efficientnet/', 'sax', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'sax'),
	(523, '/var/appdata/models/classification/efficientnet/', 'flute', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'flute'),
	(524, '/var/appdata/models/classification/efficientnet/', 'daisy', 'C', 'I', 'efficientnet', 'imagenet', '4', '43', '172', 'daisy'),
	(525, '/var/appdata/models/classification/efficientnet/', 'yellow_ladys-slipper', 'C', 'I', 'efficientnet', 'imagenet', '3', '43', '172', 'yellow_ladys-slipper'),
	(526, '/var/appdata/models/classification/efficientnet/', 'cliff', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'cliff'),
	(527, '/var/appdata/models/classification/efficientnet/', 'valley', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'valley'),
	(528, '/var/appdata/models/classification/efficientnet/', 'alp', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'alp'),
	(529, '/var/appdata/models/classification/efficientnet/', 'volcano', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'volcano'),
	(530, '/var/appdata/models/classification/efficientnet/', 'promontory', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'promontory');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(531, '/var/appdata/models/classification/efficientnet/', 'sandbar', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'sandbar'),
	(532, '/var/appdata/models/classification/efficientnet/', 'coral_reef', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'coral_reef'),
	(533, '/var/appdata/models/classification/efficientnet/', 'lakeside', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'lakeside'),
	(534, '/var/appdata/models/classification/efficientnet/', 'seashore', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'seashore'),
	(535, '/var/appdata/models/classification/efficientnet/', 'geyser', 'C', 'I', 'efficientnet', 'imagenet', '6', '6', '172', 'geyser'),
	(536, '/var/appdata/models/classification/efficientnet/', 'hatchet', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'hatchet'),
	(537, '/var/appdata/models/classification/efficientnet/', 'cleaver', 'C', 'I', 'efficientnet', 'imagenet', '3', '46', '172', 'cleaver'),
	(538, '/var/appdata/models/classification/efficientnet/', 'letter_opener', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'letter_opener'),
	(539, '/var/appdata/models/classification/efficientnet/', 'plane', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'plane'),
	(540, '/var/appdata/models/classification/efficientnet/', 'power_drill', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'power_drill');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(541, '/var/appdata/models/classification/efficientnet/', 'lawn_mower', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'lawn_mower'),
	(542, '/var/appdata/models/classification/efficientnet/', 'hammer', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'hammer'),
	(543, '/var/appdata/models/classification/efficientnet/', 'corkscrew', 'C', 'I', 'efficientnet', 'imagenet', '3', '46', '172', 'corkscrew'),
	(544, '/var/appdata/models/classification/efficientnet/', 'can_opener', 'C', 'I', 'efficientnet', 'imagenet', '3', '46', '172', 'can_opener'),
	(545, '/var/appdata/models/classification/efficientnet/', 'plunger', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'plunger'),
	(546, '/var/appdata/models/classification/efficientnet/', 'screwdriver', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'screwdriver'),
	(547, '/var/appdata/models/classification/efficientnet/', 'shovel', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'shovel'),
	(548, '/var/appdata/models/classification/efficientnet/', 'plow', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'plow'),
	(549, '/var/appdata/models/classification/efficientnet/', 'chain_saw', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'chain_saw'),
	(550, '/var/appdata/models/classification/efficientnet/', 'cock', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'cock');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(551, '/var/appdata/models/classification/efficientnet/', 'hen', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'hen'),
	(552, '/var/appdata/models/classification/efficientnet/', 'ostrich', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'ostrich'),
	(553, '/var/appdata/models/classification/efficientnet/', 'brambling', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'brambling'),
	(554, '/var/appdata/models/classification/efficientnet/', 'goldfinch', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'goldfinch'),
	(555, '/var/appdata/models/classification/efficientnet/', 'house_finch', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'house_finch'),
	(556, '/var/appdata/models/classification/efficientnet/', 'junco', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'junco'),
	(557, '/var/appdata/models/classification/efficientnet/', 'indigo_bunting', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'indigo_bunting'),
	(558, '/var/appdata/models/classification/efficientnet/', 'robin', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'robin'),
	(559, '/var/appdata/models/classification/efficientnet/', 'bulbul', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'bulbul'),
	(560, '/var/appdata/models/classification/efficientnet/', 'jay', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'jay');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(561, '/var/appdata/models/classification/efficientnet/', 'magpie', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'magpie'),
	(562, '/var/appdata/models/classification/efficientnet/', 'water_ouzel', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'water_ouzel'),
	(563, '/var/appdata/models/classification/efficientnet/', 'kite', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'kite'),
	(564, '/var/appdata/models/classification/efficientnet/', 'bald_eagle', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'bald_eagle'),
	(565, '/var/appdata/models/classification/efficientnet/', 'vulture', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'vulture'),
	(566, '/var/appdata/models/classification/efficientnet/', 'great_grey_owl', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'great_grey_owl'),
	(567, '/var/appdata/models/classification/efficientnet/', 'black_grouse', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'black_grouse'),
	(568, '/var/appdata/models/classification/efficientnet/', 'ptarmigan', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'ptarmigan'),
	(569, '/var/appdata/models/classification/efficientnet/', 'ruffed_grouse', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'ruffed_grouse'),
	(570, '/var/appdata/models/classification/efficientnet/', 'prairie_chicken', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'prairie_chicken');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(571, '/var/appdata/models/classification/efficientnet/', 'peacock', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'peacock'),
	(572, '/var/appdata/models/classification/efficientnet/', 'quail', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'quail'),
	(573, '/var/appdata/models/classification/efficientnet/', 'partridge', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'partridge'),
	(574, '/var/appdata/models/classification/efficientnet/', 'African_grey', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'African_grey'),
	(575, '/var/appdata/models/classification/efficientnet/', 'macaw', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'macaw'),
	(576, '/var/appdata/models/classification/efficientnet/', 'sulphur-crested_cockatoo', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'sulphur-crested_cockatoo'),
	(577, '/var/appdata/models/classification/efficientnet/', 'lorikeet', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'lorikeet'),
	(578, '/var/appdata/models/classification/efficientnet/', 'coucal', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'coucal'),
	(579, '/var/appdata/models/classification/efficientnet/', 'bee_eater', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'bee_eater'),
	(580, '/var/appdata/models/classification/efficientnet/', 'hornbill', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'hornbill');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(581, '/var/appdata/models/classification/efficientnet/', 'hummingbird', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'hummingbird'),
	(582, '/var/appdata/models/classification/efficientnet/', 'jacamar', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'jacamar'),
	(583, '/var/appdata/models/classification/efficientnet/', 'toucan', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'toucan'),
	(584, '/var/appdata/models/classification/efficientnet/', 'drake', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'drake'),
	(585, '/var/appdata/models/classification/efficientnet/', 'red-breasted_merganser', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'red-breasted_merganser'),
	(586, '/var/appdata/models/classification/efficientnet/', 'goose', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'goose'),
	(587, '/var/appdata/models/classification/efficientnet/', 'black_swan', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'black_swan'),
	(588, '/var/appdata/models/classification/efficientnet/', 'white_stork', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'white_stork'),
	(589, '/var/appdata/models/classification/efficientnet/', 'black_stork', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'black_stork'),
	(590, '/var/appdata/models/classification/efficientnet/', 'spoonbill', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'spoonbill');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(591, '/var/appdata/models/classification/efficientnet/', 'flamingo', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'flamingo'),
	(592, '/var/appdata/models/classification/efficientnet/', 'American_egret', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'American_egret'),
	(593, '/var/appdata/models/classification/efficientnet/', 'little_blue_heron', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'little_blue_heron'),
	(594, '/var/appdata/models/classification/efficientnet/', 'bittern', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'bittern'),
	(595, '/var/appdata/models/classification/efficientnet/', 'crane', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'crane'),
	(596, '/var/appdata/models/classification/efficientnet/', 'limpkin', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'limpkin'),
	(597, '/var/appdata/models/classification/efficientnet/', 'American_coot', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'American_coot'),
	(598, '/var/appdata/models/classification/efficientnet/', 'bustard', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'bustard'),
	(599, '/var/appdata/models/classification/efficientnet/', 'ruddy_turnstone', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'ruddy_turnstone'),
	(600, '/var/appdata/models/classification/efficientnet/', 'red-backed_sandpiper', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'red-backed_sandpiper');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(601, '/var/appdata/models/classification/efficientnet/', 'redshank', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'redshank'),
	(602, '/var/appdata/models/classification/efficientnet/', 'dowitcher', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'dowitcher'),
	(603, '/var/appdata/models/classification/efficientnet/', 'oystercatcher', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'oystercatcher'),
	(604, '/var/appdata/models/classification/efficientnet/', 'European_gallinule', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'European_gallinule'),
	(605, '/var/appdata/models/classification/efficientnet/', 'pelican', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'pelican'),
	(606, '/var/appdata/models/classification/efficientnet/', 'king_penguin', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'king_penguin'),
	(607, '/var/appdata/models/classification/efficientnet/', 'albatross', 'C', 'I', 'efficientnet', 'imagenet', '2', '47', '172', 'albatross'),
	(608, '/var/appdata/models/classification/efficientnet/', 'great_white_shark', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'great_white_shark'),
	(609, '/var/appdata/models/classification/efficientnet/', 'tiger_shark', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'tiger_shark'),
	(610, '/var/appdata/models/classification/efficientnet/', 'hammerhead', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'hammerhead');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(611, '/var/appdata/models/classification/efficientnet/', 'electric_ray', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'electric_ray'),
	(612, '/var/appdata/models/classification/efficientnet/', 'stingray', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'stingray'),
	(613, '/var/appdata/models/classification/efficientnet/', 'barracouta', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'barracouta'),
	(614, '/var/appdata/models/classification/efficientnet/', 'coho', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'coho'),
	(615, '/var/appdata/models/classification/efficientnet/', 'tench', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'tench'),
	(616, '/var/appdata/models/classification/efficientnet/', 'goldfish', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'goldfish'),
	(617, '/var/appdata/models/classification/efficientnet/', 'eel', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'eel'),
	(618, '/var/appdata/models/classification/efficientnet/', 'rock_beauty', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'rock_beauty'),
	(619, '/var/appdata/models/classification/efficientnet/', 'anemone_fish', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'anemone_fish'),
	(620, '/var/appdata/models/classification/efficientnet/', 'lionfish', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'lionfish');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(621, '/var/appdata/models/classification/efficientnet/', 'puffer', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'puffer'),
	(622, '/var/appdata/models/classification/efficientnet/', 'sturgeon', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'sturgeon'),
	(623, '/var/appdata/models/classification/efficientnet/', 'gar', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'gar'),
	(624, '/var/appdata/models/classification/efficientnet/', 'loggerhead', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'loggerhead'),
	(625, '/var/appdata/models/classification/efficientnet/', 'leatherback_turtle', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'leatherback_turtle'),
	(626, '/var/appdata/models/classification/efficientnet/', 'mud_turtle', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'mud_turtle'),
	(627, '/var/appdata/models/classification/efficientnet/', 'terrapin', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'terrapin'),
	(628, '/var/appdata/models/classification/efficientnet/', 'box_turtle', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'box_turtle'),
	(629, '/var/appdata/models/classification/efficientnet/', 'banded_gecko', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'banded_gecko'),
	(630, '/var/appdata/models/classification/efficientnet/', 'common_iguana', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'common_iguana');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(631, '/var/appdata/models/classification/efficientnet/', 'American_chameleon', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'American_chameleon'),
	(632, '/var/appdata/models/classification/efficientnet/', 'whiptail', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'whiptail'),
	(633, '/var/appdata/models/classification/efficientnet/', 'agama', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'agama'),
	(634, '/var/appdata/models/classification/efficientnet/', 'frilled_lizard', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'frilled_lizard'),
	(635, '/var/appdata/models/classification/efficientnet/', 'alligator_lizard', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'alligator_lizard'),
	(636, '/var/appdata/models/classification/efficientnet/', 'Gila_monster', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'Gila_monster'),
	(637, '/var/appdata/models/classification/efficientnet/', 'green_lizard', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'green_lizard'),
	(638, '/var/appdata/models/classification/efficientnet/', 'African_chameleon', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'African_chameleon'),
	(639, '/var/appdata/models/classification/efficientnet/', 'Komodo_dragon', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'Komodo_dragon'),
	(640, '/var/appdata/models/classification/efficientnet/', 'triceratops', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'triceratops');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(641, '/var/appdata/models/classification/efficientnet/', 'African_crocodile', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'African_crocodile'),
	(642, '/var/appdata/models/classification/efficientnet/', 'American_alligator', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'American_alligator'),
	(643, '/var/appdata/models/classification/efficientnet/', 'thunder_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'thunder_snake'),
	(644, '/var/appdata/models/classification/efficientnet/', 'ringneck_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'ringneck_snake'),
	(645, '/var/appdata/models/classification/efficientnet/', 'hognose_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'hognose_snake'),
	(646, '/var/appdata/models/classification/efficientnet/', 'green_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'green_snake'),
	(647, '/var/appdata/models/classification/efficientnet/', 'king_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'king_snake'),
	(648, '/var/appdata/models/classification/efficientnet/', 'garter_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'garter_snake'),
	(649, '/var/appdata/models/classification/efficientnet/', 'water_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'water_snake'),
	(650, '/var/appdata/models/classification/efficientnet/', 'vine_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'vine_snake');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(651, '/var/appdata/models/classification/efficientnet/', 'night_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'night_snake'),
	(652, '/var/appdata/models/classification/efficientnet/', 'boa_constrictor', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'boa_constrictor'),
	(653, '/var/appdata/models/classification/efficientnet/', 'rock_python', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'rock_python'),
	(654, '/var/appdata/models/classification/efficientnet/', 'Indian_cobra', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'Indian_cobra'),
	(655, '/var/appdata/models/classification/efficientnet/', 'green_mamba', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'green_mamba'),
	(656, '/var/appdata/models/classification/efficientnet/', 'sea_snake', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'sea_snake'),
	(657, '/var/appdata/models/classification/efficientnet/', 'horned_viper', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'horned_viper'),
	(658, '/var/appdata/models/classification/efficientnet/', 'diamondback', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'diamondback'),
	(659, '/var/appdata/models/classification/efficientnet/', 'sidewinder', 'C', 'I', 'efficientnet', 'imagenet', '2', '49', '172', 'sidewinder'),
	(660, '/var/appdata/models/classification/efficientnet/', 'European_fire_salamander', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'European_fire_salamander');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(661, '/var/appdata/models/classification/efficientnet/', 'common_newt', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'common_newt'),
	(662, '/var/appdata/models/classification/efficientnet/', 'eft', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'eft'),
	(663, '/var/appdata/models/classification/efficientnet/', 'spotted_salamander', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'spotted_salamander'),
	(664, '/var/appdata/models/classification/efficientnet/', 'axolotl', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'axolotl'),
	(665, '/var/appdata/models/classification/efficientnet/', 'bullfrog', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'bullfrog'),
	(666, '/var/appdata/models/classification/efficientnet/', 'tree_frog', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'tree_frog'),
	(667, '/var/appdata/models/classification/efficientnet/', 'tailed_frog', 'C', 'I', 'efficientnet', 'imagenet', '2', '39', '172', 'tailed_frog'),
	(668, '/var/appdata/models/classification/efficientnet/', 'whistle', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'whistle'),
	(669, '/var/appdata/models/classification/efficientnet/', 'wing', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'wing'),
	(670, '/var/appdata/models/classification/efficientnet/', 'paintbrush', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'paintbrush');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(671, '/var/appdata/models/classification/efficientnet/', 'hand_blower', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'hand_blower'),
	(672, '/var/appdata/models/classification/efficientnet/', 'oxygen_mask', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'oxygen_mask'),
	(673, '/var/appdata/models/classification/efficientnet/', 'snorkel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'snorkel'),
	(674, '/var/appdata/models/classification/efficientnet/', 'loudspeaker', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'loudspeaker'),
	(675, '/var/appdata/models/classification/efficientnet/', 'microphone', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'microphone'),
	(676, '/var/appdata/models/classification/efficientnet/', 'screen', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'screen'),
	(677, '/var/appdata/models/classification/efficientnet/', 'mouse', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'mouse'),
	(678, '/var/appdata/models/classification/efficientnet/', 'electric_fan', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'electric_fan'),
	(679, '/var/appdata/models/classification/efficientnet/', 'oil_filter', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'oil_filter'),
	(680, '/var/appdata/models/classification/efficientnet/', 'strainer', 'C', 'I', 'efficientnet', 'imagenet', '3', '46', '172', 'strainer');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(681, '/var/appdata/models/classification/efficientnet/', 'space_heater', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'space_heater'),
	(682, '/var/appdata/models/classification/efficientnet/', 'guillotine', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'guillotine'),
	(683, '/var/appdata/models/classification/efficientnet/', 'barometer', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'barometer'),
	(684, '/var/appdata/models/classification/efficientnet/', 'rule', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'rule'),
	(685, '/var/appdata/models/classification/efficientnet/', 'odometer', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'odometer'),
	(686, '/var/appdata/models/classification/efficientnet/', 'scale', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'scale'),
	(687, '/var/appdata/models/classification/efficientnet/', 'analog_clock', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'analog_clock'),
	(688, '/var/appdata/models/classification/efficientnet/', 'digital_clock', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'digital_clock'),
	(689, '/var/appdata/models/classification/efficientnet/', 'wall_clock', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'wall_clock'),
	(690, '/var/appdata/models/classification/efficientnet/', 'hourglass', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'hourglass');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(691, '/var/appdata/models/classification/efficientnet/', 'sundial', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'sundial'),
	(692, '/var/appdata/models/classification/efficientnet/', 'parking_meter', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'parking_meter'),
	(693, '/var/appdata/models/classification/efficientnet/', 'stopwatch', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'stopwatch'),
	(694, '/var/appdata/models/classification/efficientnet/', 'digital_watch', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'digital_watch'),
	(695, '/var/appdata/models/classification/efficientnet/', 'stethoscope', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'stethoscope'),
	(696, '/var/appdata/models/classification/efficientnet/', 'syringe', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'syringe'),
	(697, '/var/appdata/models/classification/efficientnet/', 'magnetic_compass', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'magnetic_compass'),
	(698, '/var/appdata/models/classification/efficientnet/', 'binoculars', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'binoculars'),
	(699, '/var/appdata/models/classification/efficientnet/', 'projector', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'projector'),
	(700, '/var/appdata/models/classification/efficientnet/', 'sunglasses', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'sunglasses');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(701, '/var/appdata/models/classification/efficientnet/', 'loupe', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'loupe'),
	(702, '/var/appdata/models/classification/efficientnet/', 'radio_telescope', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'radio_telescope'),
	(703, '/var/appdata/models/classification/efficientnet/', 'bow', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bow'),
	(704, '/var/appdata/models/classification/efficientnet/', 'cannon', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'cannon'),
	(705, '/var/appdata/models/classification/efficientnet/', 'assault_rifle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'assault_rifle'),
	(706, '/var/appdata/models/classification/efficientnet/', 'rifle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'rifle'),
	(707, '/var/appdata/models/classification/efficientnet/', 'projectile', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'projectile'),
	(708, '/var/appdata/models/classification/efficientnet/', 'computer_keyboard', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'computer_keyboard'),
	(709, '/var/appdata/models/classification/efficientnet/', 'typewriter_keyboard', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'typewriter_keyboard'),
	(710, '/var/appdata/models/classification/efficientnet/', 'crane', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'crane');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(711, '/var/appdata/models/classification/efficientnet/', 'lighter', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'lighter'),
	(712, '/var/appdata/models/classification/efficientnet/', 'abacus', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'abacus'),
	(713, '/var/appdata/models/classification/efficientnet/', 'cash_machine', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'cash_machine'),
	(714, '/var/appdata/models/classification/efficientnet/', 'slide_rule', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'slide_rule'),
	(715, '/var/appdata/models/classification/efficientnet/', 'desktop_computer', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'desktop_computer'),
	(716, '/var/appdata/models/classification/efficientnet/', 'hand-held_computer', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'hand-held_computer'),
	(717, '/var/appdata/models/classification/efficientnet/', 'notebook', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'notebook'),
	(718, '/var/appdata/models/classification/efficientnet/', 'web_site', 'C', 'I', 'efficientnet', 'imagenet', '7', '7', '172', 'web_site'),
	(719, '/var/appdata/models/classification/efficientnet/', 'harvester', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'harvester'),
	(720, '/var/appdata/models/classification/efficientnet/', 'thresher', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'thresher');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(721, '/var/appdata/models/classification/efficientnet/', 'printer', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'printer'),
	(722, '/var/appdata/models/classification/efficientnet/', 'slot', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'slot'),
	(723, '/var/appdata/models/classification/efficientnet/', 'vending_machine', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'vending_machine'),
	(724, '/var/appdata/models/classification/efficientnet/', 'sewing_machine', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'sewing_machine'),
	(725, '/var/appdata/models/classification/efficientnet/', 'joystick', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'joystick'),
	(726, '/var/appdata/models/classification/efficientnet/', 'switch', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'switch'),
	(727, '/var/appdata/models/classification/efficientnet/', 'hook', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'hook'),
	(728, '/var/appdata/models/classification/efficientnet/', 'car_wheel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'car_wheel'),
	(729, '/var/appdata/models/classification/efficientnet/', 'paddlewheel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'paddlewheel'),
	(730, '/var/appdata/models/classification/efficientnet/', 'pinwheel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pinwheel');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(731, '/var/appdata/models/classification/efficientnet/', 'potters_wheel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'potters_wheel'),
	(732, '/var/appdata/models/classification/efficientnet/', 'gas_pump', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'gas_pump'),
	(733, '/var/appdata/models/classification/efficientnet/', 'carousel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'carousel'),
	(734, '/var/appdata/models/classification/efficientnet/', 'swing', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'swing'),
	(735, '/var/appdata/models/classification/efficientnet/', 'reel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'reel'),
	(736, '/var/appdata/models/classification/efficientnet/', 'radiator', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'radiator'),
	(737, '/var/appdata/models/classification/efficientnet/', 'puck', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'puck'),
	(738, '/var/appdata/models/classification/efficientnet/', 'hard_disc', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'hard_disc'),
	(739, '/var/appdata/models/classification/efficientnet/', 'pick', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pick'),
	(740, '/var/appdata/models/classification/efficientnet/', 'car_mirror', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'car_mirror');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(741, '/var/appdata/models/classification/efficientnet/', 'solar_dish', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'solar_dish'),
	(742, '/var/appdata/models/classification/efficientnet/', 'remote_control', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'remote_control'),
	(743, '/var/appdata/models/classification/efficientnet/', 'disk_brake', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'disk_brake'),
	(744, '/var/appdata/models/classification/efficientnet/', 'buckle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'buckle'),
	(745, '/var/appdata/models/classification/efficientnet/', 'hair_slide', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'hair_slide'),
	(746, '/var/appdata/models/classification/efficientnet/', 'knot', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'knot'),
	(747, '/var/appdata/models/classification/efficientnet/', 'combination_lock', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'combination_lock'),
	(748, '/var/appdata/models/classification/efficientnet/', 'padlock', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'padlock'),
	(749, '/var/appdata/models/classification/efficientnet/', 'nail', 'C', 'I', 'efficientnet', 'imagenet', '7', '42', '172', 'nail'),
	(750, '/var/appdata/models/classification/efficientnet/', 'safety_pin', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'safety_pin');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(751, '/var/appdata/models/classification/efficientnet/', 'screw', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'screw'),
	(752, '/var/appdata/models/classification/efficientnet/', 'muzzle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'muzzle'),
	(753, '/var/appdata/models/classification/efficientnet/', 'seat_belt', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'seat_belt'),
	(754, '/var/appdata/models/classification/efficientnet/', 'ski', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'ski'),
	(755, '/var/appdata/models/classification/efficientnet/', 'candle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'candle'),
	(756, '/var/appdata/models/classification/efficientnet/', 'jack-o-lantern', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'jack-o-lantern'),
	(757, '/var/appdata/models/classification/efficientnet/', 'spotlight', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'spotlight'),
	(758, '/var/appdata/models/classification/efficientnet/', 'torch', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'torch'),
	(759, '/var/appdata/models/classification/efficientnet/', 'neck_brace', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'neck_brace'),
	(760, '/var/appdata/models/classification/efficientnet/', 'pier', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pier');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(761, '/var/appdata/models/classification/efficientnet/', 'maypole', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'maypole'),
	(762, '/var/appdata/models/classification/efficientnet/', 'mousetrap', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'mousetrap'),
	(763, '/var/appdata/models/classification/efficientnet/', 'spider_web', 'C', 'I', 'efficientnet', 'imagenet', '7', '7', '172', 'spider_web'),
	(764, '/var/appdata/models/classification/efficientnet/', 'trilobite', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'trilobite'),
	(765, '/var/appdata/models/classification/efficientnet/', 'harvestman', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'harvestman'),
	(766, '/var/appdata/models/classification/efficientnet/', 'scorpion', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'scorpion'),
	(767, '/var/appdata/models/classification/efficientnet/', 'black_and_gold_garden_spider', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'black_and_gold_garden_spider'),
	(768, '/var/appdata/models/classification/efficientnet/', 'barn_spider', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'barn_spider'),
	(769, '/var/appdata/models/classification/efficientnet/', 'garden_spider', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'garden_spider'),
	(770, '/var/appdata/models/classification/efficientnet/', 'black_widow', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'black_widow');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(771, '/var/appdata/models/classification/efficientnet/', 'tarantula', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'tarantula'),
	(772, '/var/appdata/models/classification/efficientnet/', 'wolf_spider', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'wolf_spider'),
	(773, '/var/appdata/models/classification/efficientnet/', 'tick', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'tick'),
	(774, '/var/appdata/models/classification/efficientnet/', 'centipede', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'centipede'),
	(775, '/var/appdata/models/classification/efficientnet/', 'isop', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'isop'),
	(776, '/var/appdata/models/classification/efficientnet/', 'Dungeness_crab', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'Dungeness_crab'),
	(777, '/var/appdata/models/classification/efficientnet/', 'rock_crab', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'rock_crab'),
	(778, '/var/appdata/models/classification/efficientnet/', 'fiddler_crab', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'fiddler_crab'),
	(779, '/var/appdata/models/classification/efficientnet/', 'king_crab', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'king_crab'),
	(780, '/var/appdata/models/classification/efficientnet/', 'American_lobster', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'American_lobster');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(781, '/var/appdata/models/classification/efficientnet/', 'spiny_lobster', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'spiny_lobster'),
	(782, '/var/appdata/models/classification/efficientnet/', 'crayfish', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'crayfish'),
	(783, '/var/appdata/models/classification/efficientnet/', 'hermit_crab', 'C', 'I', 'efficientnet', 'imagenet', '2', '51', '172', 'hermit_crab'),
	(784, '/var/appdata/models/classification/efficientnet/', 'tiger_beetle', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'tiger_beetle'),
	(785, '/var/appdata/models/classification/efficientnet/', 'ladybug', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'ladybug'),
	(786, '/var/appdata/models/classification/efficientnet/', 'ground_beetle', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'ground_beetle'),
	(787, '/var/appdata/models/classification/efficientnet/', 'long-horned_beetle', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'long-horned_beetle'),
	(788, '/var/appdata/models/classification/efficientnet/', 'leaf_beetle', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'leaf_beetle'),
	(789, '/var/appdata/models/classification/efficientnet/', 'dung_beetle', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'dung_beetle'),
	(790, '/var/appdata/models/classification/efficientnet/', 'rhinoceros_beetle', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'rhinoceros_beetle');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(791, '/var/appdata/models/classification/efficientnet/', 'weevil', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'weevil'),
	(792, '/var/appdata/models/classification/efficientnet/', 'fly', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'fly'),
	(793, '/var/appdata/models/classification/efficientnet/', 'bee', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'bee'),
	(794, '/var/appdata/models/classification/efficientnet/', 'grasshopper', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'grasshopper'),
	(795, '/var/appdata/models/classification/efficientnet/', 'cricket', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'cricket'),
	(796, '/var/appdata/models/classification/efficientnet/', 'walking_stick', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'walking_stick'),
	(797, '/var/appdata/models/classification/efficientnet/', 'cockroach', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'cockroach'),
	(798, '/var/appdata/models/classification/efficientnet/', 'mantis', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'mantis'),
	(799, '/var/appdata/models/classification/efficientnet/', 'cicada', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'cicada'),
	(800, '/var/appdata/models/classification/efficientnet/', 'leafhopper', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'leafhopper');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(801, '/var/appdata/models/classification/efficientnet/', 'lacewing', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'lacewing'),
	(802, '/var/appdata/models/classification/efficientnet/', 'dragonfly', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'dragonfly'),
	(803, '/var/appdata/models/classification/efficientnet/', 'damselfly', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'damselfly'),
	(804, '/var/appdata/models/classification/efficientnet/', 'admiral', 'C', 'I', 'efficientnet', 'imagenet', '1', '1', '172', 'admiral'),
	(805, '/var/appdata/models/classification/efficientnet/', 'ringlet', 'C', 'I', 'efficientnet', 'imagenet', '7', '42', '172', 'ringlet'),
	(806, '/var/appdata/models/classification/efficientnet/', 'monarch', 'C', 'I', 'efficientnet', 'imagenet', '1', '1', '172', 'monarch'),
	(807, '/var/appdata/models/classification/efficientnet/', 'cabbage_butterfly', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'cabbage_butterfly'),
	(808, '/var/appdata/models/classification/efficientnet/', 'sulphur_butterfly', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'sulphur_butterfly'),
	(809, '/var/appdata/models/classification/efficientnet/', 'lycaenid', 'C', 'I', 'efficientnet', 'imagenet', '2', '34', '172', 'lycaenid'),
	(810, '/var/appdata/models/classification/efficientnet/', 'jellyfish', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'jellyfish');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(811, '/var/appdata/models/classification/efficientnet/', 'sea_anemone', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'sea_anemone'),
	(812, '/var/appdata/models/classification/efficientnet/', 'brain_coral', 'C', 'I', 'efficientnet', 'imagenet', '2', '48', '172', 'brain_coral'),
	(813, '/var/appdata/models/classification/efficientnet/', 'flatworm', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'flatworm'),
	(814, '/var/appdata/models/classification/efficientnet/', 'nematode', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'nematode'),
	(815, '/var/appdata/models/classification/efficientnet/', 'conch', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'conch'),
	(816, '/var/appdata/models/classification/efficientnet/', 'snail', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'snail'),
	(817, '/var/appdata/models/classification/efficientnet/', 'slug', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'slug'),
	(818, '/var/appdata/models/classification/efficientnet/', 'sea_slug', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'sea_slug'),
	(819, '/var/appdata/models/classification/efficientnet/', 'chiton', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'chiton'),
	(820, '/var/appdata/models/classification/efficientnet/', 'sea_urchin', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'sea_urchin');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(821, '/var/appdata/models/classification/efficientnet/', 'sea_cucumber', 'C', 'I', 'efficientnet', 'imagenet', '2', '7', '172', 'sea_cucumber'),
	(822, '/var/appdata/models/classification/efficientnet/', 'iron', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'iron'),
	(823, '/var/appdata/models/classification/efficientnet/', 'espresso_maker', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'espresso_maker'),
	(824, '/var/appdata/models/classification/efficientnet/', 'microwave', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'microwave'),
	(825, '/var/appdata/models/classification/efficientnet/', 'Dutch_oven', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'Dutch_oven'),
	(826, '/var/appdata/models/classification/efficientnet/', 'rotisserie', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'rotisserie'),
	(827, '/var/appdata/models/classification/efficientnet/', 'toaster', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'toaster'),
	(828, '/var/appdata/models/classification/efficientnet/', 'waffle_iron', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'waffle_iron'),
	(829, '/var/appdata/models/classification/efficientnet/', 'vacuum', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'vacuum'),
	(830, '/var/appdata/models/classification/efficientnet/', 'dishwasher', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'dishwasher');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(831, '/var/appdata/models/classification/efficientnet/', 'refrigerator', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'refrigerator'),
	(832, '/var/appdata/models/classification/efficientnet/', 'washer', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'washer'),
	(833, '/var/appdata/models/classification/efficientnet/', 'Crock_Pot', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'Crock_Pot'),
	(834, '/var/appdata/models/classification/efficientnet/', 'frying_pan', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'frying_pan'),
	(835, '/var/appdata/models/classification/efficientnet/', 'wok', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'wok'),
	(836, '/var/appdata/models/classification/efficientnet/', 'caldron', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'caldron'),
	(837, '/var/appdata/models/classification/efficientnet/', 'coffeepot', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'coffeepot'),
	(838, '/var/appdata/models/classification/efficientnet/', 'teapot', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'teapot'),
	(839, '/var/appdata/models/classification/efficientnet/', 'spatula', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'spatula'),
	(840, '/var/appdata/models/classification/efficientnet/', 'altar', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'altar');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(841, '/var/appdata/models/classification/efficientnet/', 'triumphal_arch', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'triumphal_arch'),
	(842, '/var/appdata/models/classification/efficientnet/', 'patio', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'patio'),
	(843, '/var/appdata/models/classification/efficientnet/', 'steel_arch_bridge', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'steel_arch_bridge'),
	(844, '/var/appdata/models/classification/efficientnet/', 'suspension_bridge', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'suspension_bridge'),
	(845, '/var/appdata/models/classification/efficientnet/', 'viaduct', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'viaduct'),
	(846, '/var/appdata/models/classification/efficientnet/', 'barn', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'barn'),
	(847, '/var/appdata/models/classification/efficientnet/', 'greenhouse', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'greenhouse'),
	(848, '/var/appdata/models/classification/efficientnet/', 'palace', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'palace'),
	(849, '/var/appdata/models/classification/efficientnet/', 'monastery', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'monastery'),
	(850, '/var/appdata/models/classification/efficientnet/', 'library', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'library');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(851, '/var/appdata/models/classification/efficientnet/', 'apiary', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'apiary'),
	(852, '/var/appdata/models/classification/efficientnet/', 'boathouse', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'boathouse'),
	(853, '/var/appdata/models/classification/efficientnet/', 'church', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'church'),
	(854, '/var/appdata/models/classification/efficientnet/', 'mosque', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'mosque'),
	(855, '/var/appdata/models/classification/efficientnet/', 'stupa', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'stupa'),
	(856, '/var/appdata/models/classification/efficientnet/', 'planetarium', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'planetarium'),
	(857, '/var/appdata/models/classification/efficientnet/', 'restaurant', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'restaurant'),
	(858, '/var/appdata/models/classification/efficientnet/', 'cinema', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'cinema'),
	(859, '/var/appdata/models/classification/efficientnet/', 'home_theater', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'home_theater'),
	(860, '/var/appdata/models/classification/efficientnet/', 'lumbermill', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'lumbermill');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(861, '/var/appdata/models/classification/efficientnet/', 'coil', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'coil'),
	(862, '/var/appdata/models/classification/efficientnet/', 'obelisk', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'obelisk'),
	(863, '/var/appdata/models/classification/efficientnet/', 'totem_pole', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'totem_pole'),
	(864, '/var/appdata/models/classification/efficientnet/', 'castle', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'castle'),
	(865, '/var/appdata/models/classification/efficientnet/', 'prison', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'prison'),
	(866, '/var/appdata/models/classification/efficientnet/', 'grocery_store', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'grocery_store'),
	(867, '/var/appdata/models/classification/efficientnet/', 'bakery', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'bakery'),
	(868, '/var/appdata/models/classification/efficientnet/', 'barbershop', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'barbershop'),
	(869, '/var/appdata/models/classification/efficientnet/', 'bookshop', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'bookshop'),
	(870, '/var/appdata/models/classification/efficientnet/', 'confectionery', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'confectionery');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(871, '/var/appdata/models/classification/efficientnet/', 'shoe_shop', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'shoe_shop'),
	(872, '/var/appdata/models/classification/efficientnet/', 'tobacco_shop', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'tobacco_shop'),
	(873, '/var/appdata/models/classification/efficientnet/', 'toyshop', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'toyshop'),
	(874, '/var/appdata/models/classification/efficientnet/', 'fountain', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'fountain'),
	(875, '/var/appdata/models/classification/efficientnet/', 'cliff_dwelling', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'cliff_dwelling'),
	(876, '/var/appdata/models/classification/efficientnet/', 'yurt', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'yurt'),
	(877, '/var/appdata/models/classification/efficientnet/', 'dock', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'dock'),
	(878, '/var/appdata/models/classification/efficientnet/', 'megalith', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'megalith'),
	(879, '/var/appdata/models/classification/efficientnet/', 'megalith', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'megalith'),
	(880, '/var/appdata/models/classification/efficientnet/', 'bannister', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'bannister');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(881, '/var/appdata/models/classification/efficientnet/', 'breakwater', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'breakwater'),
	(882, '/var/appdata/models/classification/efficientnet/', 'dam', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'dam'),
	(883, '/var/appdata/models/classification/efficientnet/', 'chainlink_fence', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'chainlink_fence'),
	(884, '/var/appdata/models/classification/efficientnet/', 'picket_fence', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'picket_fence'),
	(885, '/var/appdata/models/classification/efficientnet/', 'worm_fence', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'worm_fence'),
	(886, '/var/appdata/models/classification/efficientnet/', 'stone_wall', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'stone_wall'),
	(887, '/var/appdata/models/classification/efficientnet/', 'grille', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'grille'),
	(888, '/var/appdata/models/classification/efficientnet/', 'sliding_door', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'sliding_door'),
	(889, '/var/appdata/models/classification/efficientnet/', 'turnstile', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'turnstile'),
	(890, '/var/appdata/models/classification/efficientnet/', 'mountain_tent', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'mountain_tent');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(891, '/var/appdata/models/classification/efficientnet/', 'scoreboard', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'scoreboard'),
	(892, '/var/appdata/models/classification/efficientnet/', 'honeycomb', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'honeycomb'),
	(893, '/var/appdata/models/classification/efficientnet/', 'plate_rack', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'plate_rack'),
	(894, '/var/appdata/models/classification/efficientnet/', 'pedestal', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pedestal'),
	(895, '/var/appdata/models/classification/efficientnet/', 'stone_wall', 'C', 'I', 'efficientnet', 'imagenet', '3', '41', '172', 'stone_wall'),
	(896, '/var/appdata/models/classification/efficientnet/', 'mashed_potato', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'mashed_potato'),
	(897, '/var/appdata/models/classification/efficientnet/', 'bell_pepper', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'bell_pepper'),
	(898, '/var/appdata/models/classification/efficientnet/', 'head_cabbage', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'head_cabbage'),
	(899, '/var/appdata/models/classification/efficientnet/', 'broccoli', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'broccoli'),
	(900, '/var/appdata/models/classification/efficientnet/', 'cauliflower', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'cauliflower');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(901, '/var/appdata/models/classification/efficientnet/', 'zucchini', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'zucchini'),
	(902, '/var/appdata/models/classification/efficientnet/', 'spaghetti_squash', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'spaghetti_squash'),
	(903, '/var/appdata/models/classification/efficientnet/', 'acorn_squash', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'acorn_squash'),
	(904, '/var/appdata/models/classification/efficientnet/', 'butternut_squash', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'butternut_squash'),
	(905, '/var/appdata/models/classification/efficientnet/', 'cucumber', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'cucumber'),
	(906, '/var/appdata/models/classification/efficientnet/', 'artichoke', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'artichoke'),
	(907, '/var/appdata/models/classification/efficientnet/', 'cardoon', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'cardoon'),
	(908, '/var/appdata/models/classification/efficientnet/', 'mushroom', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'mushroom'),
	(909, '/var/appdata/models/classification/efficientnet/', 'shower_curtain', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'shower_curtain'),
	(910, '/var/appdata/models/classification/efficientnet/', 'jean', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'jean');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(911, '/var/appdata/models/classification/efficientnet/', 'carton', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'carton'),
	(912, '/var/appdata/models/classification/efficientnet/', 'handkerchief', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'handkerchief'),
	(913, '/var/appdata/models/classification/efficientnet/', 'sandal', 'C', 'I', 'efficientnet', 'imagenet', '3', '60', '172', 'sandal'),
	(914, '/var/appdata/models/classification/efficientnet/', 'ashcan', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'ashcan'),
	(915, '/var/appdata/models/classification/efficientnet/', 'safe', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'safe'),
	(916, '/var/appdata/models/classification/efficientnet/', 'plate', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'plate'),
	(917, '/var/appdata/models/classification/efficientnet/', 'necklace', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'necklace'),
	(918, '/var/appdata/models/classification/efficientnet/', 'croquet_ball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'croquet_ball'),
	(919, '/var/appdata/models/classification/efficientnet/', 'fur_coat', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'fur_coat'),
	(920, '/var/appdata/models/classification/efficientnet/', 'thimble', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'thimble');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(921, '/var/appdata/models/classification/efficientnet/', 'pajama', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'pajama'),
	(922, '/var/appdata/models/classification/efficientnet/', 'running_shoe', 'C', 'I', 'efficientnet', 'imagenet', '3', '60', '172', 'running_shoe'),
	(923, '/var/appdata/models/classification/efficientnet/', 'cocktail_shaker', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'cocktail_shaker'),
	(924, '/var/appdata/models/classification/efficientnet/', 'chest', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'chest'),
	(925, '/var/appdata/models/classification/efficientnet/', 'manhole_cover', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'manhole_cover'),
	(926, '/var/appdata/models/classification/efficientnet/', 'modem', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'modem'),
	(927, '/var/appdata/models/classification/efficientnet/', 'tub', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'tub'),
	(928, '/var/appdata/models/classification/efficientnet/', 'tray', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'tray'),
	(929, '/var/appdata/models/classification/efficientnet/', 'balance_beam', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'balance_beam'),
	(930, '/var/appdata/models/classification/efficientnet/', 'bagel', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'bagel');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(931, '/var/appdata/models/classification/efficientnet/', 'prayer_rug', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'prayer_rug'),
	(932, '/var/appdata/models/classification/efficientnet/', 'kimono', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'kimono'),
	(933, '/var/appdata/models/classification/efficientnet/', 'hot_pot', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'hot_pot'),
	(934, '/var/appdata/models/classification/efficientnet/', 'whiskey_jug', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'whiskey_jug'),
	(935, '/var/appdata/models/classification/efficientnet/', 'knee_pad', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'knee_pad'),
	(936, '/var/appdata/models/classification/efficientnet/', 'book_jacket', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'book_jacket'),
	(937, '/var/appdata/models/classification/efficientnet/', 'spindle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'spindle'),
	(938, '/var/appdata/models/classification/efficientnet/', 'ski_mask', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'ski_mask'),
	(939, '/var/appdata/models/classification/efficientnet/', 'beer_bottle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'beer_bottle'),
	(940, '/var/appdata/models/classification/efficientnet/', 'crash_helmet', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'crash_helmet');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(941, '/var/appdata/models/classification/efficientnet/', 'bottlecap', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bottlecap'),
	(942, '/var/appdata/models/classification/efficientnet/', 'tile_roof', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'tile_roof'),
	(943, '/var/appdata/models/classification/efficientnet/', 'mask', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'mask'),
	(944, '/var/appdata/models/classification/efficientnet/', 'maillot', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'maillot'),
	(945, '/var/appdata/models/classification/efficientnet/', 'Petri_dish', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'Petri_dish'),
	(946, '/var/appdata/models/classification/efficientnet/', 'football_helmet', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'football_helmet'),
	(947, '/var/appdata/models/classification/efficientnet/', 'book_jacket', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'book_jacket'),
	(948, '/var/appdata/models/classification/efficientnet/', 'teddy', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'teddy'),
	(949, '/var/appdata/models/classification/efficientnet/', 'holster', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'holster'),
	(950, '/var/appdata/models/classification/efficientnet/', 'pop bottle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pop bottle');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(951, '/var/appdata/models/classification/efficientnet/', 'photocopier', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'photocopier'),
	(952, '/var/appdata/models/classification/efficientnet/', 'vestment', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'vestment'),
	(953, '/var/appdata/models/classification/efficientnet/', 'crossword_puzzle', 'C', 'I', 'efficientnet', 'imagenet', '7', '7', '172', 'crossword_puzzle'),
	(954, '/var/appdata/models/classification/efficientnet/', 'golf_ball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'golf_ball'),
	(955, '/var/appdata/models/classification/efficientnet/', 'trifle', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'trifle'),
	(956, '/var/appdata/models/classification/efficientnet/', 'suit', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'suit'),
	(957, '/var/appdata/models/classification/efficientnet/', 'water_tower', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'water_tower'),
	(958, '/var/appdata/models/classification/efficientnet/', 'feather_boa', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'feather_boa'),
	(959, '/var/appdata/models/classification/efficientnet/', 'cloak', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'cloak'),
	(960, '/var/appdata/models/classification/efficientnet/', 'red_wine', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'red_wine');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(961, '/var/appdata/models/classification/efficientnet/', 'drumstick', 'C', 'I', 'efficientnet', 'imagenet', '3', '33', '172', 'drumstick'),
	(962, '/var/appdata/models/classification/efficientnet/', 'shield', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'shield'),
	(963, '/var/appdata/models/classification/efficientnet/', 'Christmas_stocking', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'Christmas_stocking'),
	(964, '/var/appdata/models/classification/efficientnet/', 'hoopskirt', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'hoopskirt'),
	(965, '/var/appdata/models/classification/efficientnet/', 'menu', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'menu'),
	(966, '/var/appdata/models/classification/efficientnet/', 'stage', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'stage'),
	(967, '/var/appdata/models/classification/efficientnet/', 'bonnet', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bonnet'),
	(968, '/var/appdata/models/classification/efficientnet/', 'meat_loaf', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'meat_loaf'),
	(969, '/var/appdata/models/classification/efficientnet/', 'baseball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'baseball'),
	(970, '/var/appdata/models/classification/efficientnet/', 'face_powder', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'face_powder');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(971, '/var/appdata/models/classification/efficientnet/', 'scabbard', 'C', 'I', 'efficientnet', 'imagenet', '7', '7', '172', 'scabbard'),
	(972, '/var/appdata/models/classification/efficientnet/', 'sunscreen', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'sunscreen'),
	(973, '/var/appdata/models/classification/efficientnet/', 'beer_glass', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'beer_glass'),
	(974, '/var/appdata/models/classification/efficientnet/', 'hen-of-the-woods', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'hen-of-the-woods'),
	(975, '/var/appdata/models/classification/efficientnet/', 'hoopskirt', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'hoopskirt'),
	(976, '/var/appdata/models/classification/efficientnet/', 'lampshade', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'lampshade'),
	(977, '/var/appdata/models/classification/efficientnet/', 'wool', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'wool'),
	(978, '/var/appdata/models/classification/efficientnet/', 'hay', 'C', 'I', 'efficientnet', 'imagenet', '4', '7', '172', 'hay'),
	(979, '/var/appdata/models/classification/efficientnet/', 'bow_tie', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'bow_tie'),
	(980, '/var/appdata/models/classification/efficientnet/', 'mailbag', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'mailbag');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(981, '/var/appdata/models/classification/efficientnet/', 'water_jug', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'water_jug'),
	(982, '/var/appdata/models/classification/efficientnet/', 'bucket', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bucket'),
	(983, '/var/appdata/models/classification/efficientnet/', 'dishrag', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'dishrag'),
	(984, '/var/appdata/models/classification/efficientnet/', 'soup_bowl', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'soup_bowl'),
	(985, '/var/appdata/models/classification/efficientnet/', 'eggnog', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'eggnog'),
	(986, '/var/appdata/models/classification/efficientnet/', 'hoopskirt', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'hoopskirt'),
	(987, '/var/appdata/models/classification/efficientnet/', 'trench_coat', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'trench_coat'),
	(988, '/var/appdata/models/classification/efficientnet/', 'paddle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'paddle'),
	(989, '/var/appdata/models/classification/efficientnet/', 'chain', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'chain'),
	(990, '/var/appdata/models/classification/efficientnet/', 'swab', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'swab');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(991, '/var/appdata/models/classification/efficientnet/', 'mixing_bowl', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'mixing_bowl'),
	(992, '/var/appdata/models/classification/efficientnet/', 'potpie', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'potpie'),
	(993, '/var/appdata/models/classification/efficientnet/', 'wine_bottle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'wine_bottle'),
	(994, '/var/appdata/models/classification/efficientnet/', 'shoji', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'shoji'),
	(995, '/var/appdata/models/classification/efficientnet/', 'bulletproof_vest', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'bulletproof_vest'),
	(996, '/var/appdata/models/classification/efficientnet/', 'drilling_platform', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'drilling_platform'),
	(997, '/var/appdata/models/classification/efficientnet/', 'binder', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'binder'),
	(998, '/var/appdata/models/classification/efficientnet/', 'cardigan', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'cardigan'),
	(999, '/var/appdata/models/classification/efficientnet/', 'sweatshirt', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'sweatshirt'),
	(1000, '/var/appdata/models/classification/efficientnet/', 'pot', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'pot');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1001, '/var/appdata/models/classification/efficientnet/', 'birdhouse', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'birdhouse'),
	(1002, '/var/appdata/models/classification/efficientnet/', 'hamper', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'hamper'),
	(1003, '/var/appdata/models/classification/efficientnet/', 'ping-pong_ball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'ping-pong_ball'),
	(1004, '/var/appdata/models/classification/efficientnet/', 'pencil_box', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pencil_box'),
	(1005, '/var/appdata/models/classification/efficientnet/', 'pay-phone', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pay-phone'),
	(1006, '/var/appdata/models/classification/efficientnet/', 'consomme', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'consomme'),
	(1007, '/var/appdata/models/classification/efficientnet/', 'apron', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'apron'),
	(1008, '/var/appdata/models/classification/efficientnet/', 'punching_bag', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'punching_bag'),
	(1009, '/var/appdata/models/classification/efficientnet/', 'backpack', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'backpack'),
	(1010, '/var/appdata/models/classification/efficientnet/', 'groom', 'C', 'I', 'efficientnet', 'imagenet', '1', '1', '172', 'groom');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1011, '/var/appdata/models/classification/efficientnet/', 'bearskin', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bearskin'),
	(1012, '/var/appdata/models/classification/efficientnet/', 'pencil_sharpener', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pencil_sharpener'),
	(1013, '/var/appdata/models/classification/efficientnet/', 'broom', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'broom'),
	(1014, '/var/appdata/models/classification/efficientnet/', 'mosquito_net', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'mosquito_net'),
	(1015, '/var/appdata/models/classification/efficientnet/', 'abaya', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'abaya'),
	(1016, '/var/appdata/models/classification/efficientnet/', 'mortarboard', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'mortarboard'),
	(1017, '/var/appdata/models/classification/efficientnet/', 'poncho', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'poncho'),
	(1018, '/var/appdata/models/classification/efficientnet/', 'crutch', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'crutch'),
	(1019, '/var/appdata/models/classification/efficientnet/', 'Polaroid_camera', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'Polaroid_camera'),
	(1020, '/var/appdata/models/classification/efficientnet/', 'space_bar', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'space_bar');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1021, '/var/appdata/models/classification/efficientnet/', 'cup', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'cup'),
	(1022, '/var/appdata/models/classification/efficientnet/', 'racket', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'racket'),
	(1023, '/var/appdata/models/classification/efficientnet/', 'traffic_light', 'C', 'I', 'efficientnet', 'imagenet', '3', '41', '172', 'traffic_light'),
	(1024, '/var/appdata/models/classification/efficientnet/', 'quill', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'quill'),
	(1025, '/var/appdata/models/classification/efficientnet/', 'radio', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'radio'),
	(1026, '/var/appdata/models/classification/efficientnet/', 'dough', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'dough'),
	(1027, '/var/appdata/models/classification/efficientnet/', 'cuirass', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'cuirass'),
	(1028, '/var/appdata/models/classification/efficientnet/', 'military_uniform', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'military_uniform'),
	(1029, '/var/appdata/models/classification/efficientnet/', 'lipstick', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'lipstick'),
	(1030, '/var/appdata/models/classification/efficientnet/', 'shower_cap', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'shower_cap');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1031, '/var/appdata/models/classification/efficientnet/', 'monitor', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'monitor'),
	(1032, '/var/appdata/models/classification/efficientnet/', 'oscilloscope', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'oscilloscope'),
	(1033, '/var/appdata/models/classification/efficientnet/', 'mitten', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'mitten'),
	(1034, '/var/appdata/models/classification/efficientnet/', 'brassiere', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'brassiere'),
	(1035, '/var/appdata/models/classification/efficientnet/', 'French_loaf', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'French_loaf'),
	(1036, '/var/appdata/models/classification/efficientnet/', 'vase', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'vase'),
	(1037, '/var/appdata/models/classification/efficientnet/', 'milk_can', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'milk_can'),
	(1038, '/var/appdata/models/classification/efficientnet/', 'dough', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'dough'),
	(1039, '/var/appdata/models/classification/efficientnet/', 'rugby_ball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'rugby_ball'),
	(1040, '/var/appdata/models/classification/efficientnet/', 'paper_towel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'paper_towel');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1041, '/var/appdata/models/classification/efficientnet/', 'earthstar', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'earthstar'),
	(1042, '/var/appdata/models/classification/efficientnet/', 'envelope', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'envelope'),
	(1043, '/var/appdata/models/classification/efficientnet/', 'miniskirt', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'miniskirt'),
	(1044, '/var/appdata/models/classification/efficientnet/', 'cowboy_hat', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'cowboy_hat'),
	(1045, '/var/appdata/models/classification/efficientnet/', 'trolleybus', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'trolleybus'),
	(1046, '/var/appdata/models/classification/efficientnet/', 'perfume', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'perfume'),
	(1047, '/var/appdata/models/classification/efficientnet/', 'bathtub', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bathtub'),
	(1048, '/var/appdata/models/classification/efficientnet/', 'hotdog', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'hotdog'),
	(1049, '/var/appdata/models/classification/efficientnet/', 'coral_fungus', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'coral_fungus'),
	(1050, '/var/appdata/models/classification/efficientnet/', 'bullet_train', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'bullet_train');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1051, '/var/appdata/models/classification/efficientnet/', 'pillow', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pillow'),
	(1052, '/var/appdata/models/classification/efficientnet/', 'toilet_tissue', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'toilet_tissue'),
	(1053, '/var/appdata/models/classification/efficientnet/', 'cassette', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'cassette'),
	(1054, '/var/appdata/models/classification/efficientnet/', 'carpenters_kit', 'C', 'I', 'efficientnet', 'imagenet', '3', '45', '172', 'carpenters_kit'),
	(1055, '/var/appdata/models/classification/efficientnet/', 'ladle', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'ladle'),
	(1056, '/var/appdata/models/classification/efficientnet/', 'stinkhorn', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'stinkhorn'),
	(1057, '/var/appdata/models/classification/efficientnet/', 'lotion', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'lotion'),
	(1058, '/var/appdata/models/classification/efficientnet/', 'hair_spray', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'hair_spray'),
	(1059, '/var/appdata/models/classification/efficientnet/', 'academic_gown', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'academic_gown'),
	(1060, '/var/appdata/models/classification/efficientnet/', 'dome', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'dome');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1061, '/var/appdata/models/classification/efficientnet/', 'crate', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'crate'),
	(1062, '/var/appdata/models/classification/efficientnet/', 'wig', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'wig'),
	(1063, '/var/appdata/models/classification/efficientnet/', 'burrito', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'burrito'),
	(1064, '/var/appdata/models/classification/efficientnet/', 'pill_bottle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pill_bottle'),
	(1065, '/var/appdata/models/classification/efficientnet/', 'chain_mail', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'chain_mail'),
	(1066, '/var/appdata/models/classification/efficientnet/', 'theater_curtain', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'theater_curtain'),
	(1067, '/var/appdata/models/classification/efficientnet/', 'window_shade', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'window_shade'),
	(1068, '/var/appdata/models/classification/efficientnet/', 'barrel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'barrel'),
	(1069, '/var/appdata/models/classification/efficientnet/', 'washbasin', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'washbasin'),
	(1070, '/var/appdata/models/classification/efficientnet/', 'ballpoint', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'ballpoint');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1071, '/var/appdata/models/classification/efficientnet/', 'basketball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'basketball'),
	(1072, '/var/appdata/models/classification/efficientnet/', 'bath_towel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bath_towel'),
	(1073, '/var/appdata/models/classification/efficientnet/', 'cowboy_boot', 'C', 'I', 'efficientnet', 'imagenet', '3', '60', '172', 'cowboy_boot'),
	(1074, '/var/appdata/models/classification/efficientnet/', 'gown', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'gown'),
	(1075, '/var/appdata/models/classification/efficientnet/', 'window_screen', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'window_screen'),
	(1076, '/var/appdata/models/classification/efficientnet/', 'agaric', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'agaric'),
	(1077, '/var/appdata/models/classification/efficientnet/', 'cellular_telephone', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'cellular_telephone'),
	(1078, '/var/appdata/models/classification/efficientnet/', 'nipple', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'nipple'),
	(1079, '/var/appdata/models/classification/efficientnet/', 'barbell', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'barbell'),
	(1080, '/var/appdata/models/classification/efficientnet/', 'mailbox', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'mailbox');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1081, '/var/appdata/models/classification/efficientnet/', 'lab_coat', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'lab_coat'),
	(1082, '/var/appdata/models/classification/efficientnet/', 'fire_screen', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'fire_screen'),
	(1083, '/var/appdata/models/classification/efficientnet/', 'minibus', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'minibus'),
	(1084, '/var/appdata/models/classification/efficientnet/', 'packet', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'packet'),
	(1085, '/var/appdata/models/classification/efficientnet/', 'maze', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'maze'),
	(1086, '/var/appdata/models/classification/efficientnet/', 'pole', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'pole'),
	(1087, '/var/appdata/models/classification/efficientnet/', 'horizontal_bar', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'horizontal_bar'),
	(1088, '/var/appdata/models/classification/efficientnet/', 'sombrero', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'sombrero'),
	(1089, '/var/appdata/models/classification/efficientnet/', 'pickelhaube', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'pickelhaube'),
	(1090, '/var/appdata/models/classification/efficientnet/', 'rain_barrel', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'rain_barrel');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1091, '/var/appdata/models/classification/efficientnet/', 'wallet', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'wallet'),
	(1092, '/var/appdata/models/classification/efficientnet/', 'cassette_player', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'cassette_player'),
	(1093, '/var/appdata/models/classification/efficientnet/', 'comic_book', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'comic_book'),
	(1094, '/var/appdata/models/classification/efficientnet/', 'piggy_bank', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'piggy_bank'),
	(1095, '/var/appdata/models/classification/efficientnet/', 'street_sign', 'C', 'I', 'efficientnet', 'imagenet', '3', '41', '172', 'street_sign'),
	(1096, '/var/appdata/models/classification/efficientnet/', 'bell_cote', 'C', 'I', 'efficientnet', 'imagenet', '3', '56', '172', 'bell_cote'),
	(1097, '/var/appdata/models/classification/efficientnet/', 'fountain_pen', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'fountain_pen'),
	(1098, '/var/appdata/models/classification/efficientnet/', 'Windsor_tie', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'Windsor_tie'),
	(1099, '/var/appdata/models/classification/efficientnet/', 'volleyball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'volleyball'),
	(1100, '/var/appdata/models/classification/efficientnet/', 'overskirt', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'overskirt');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1101, '/var/appdata/models/classification/efficientnet/', 'sarong', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'sarong'),
	(1102, '/var/appdata/models/classification/efficientnet/', 'purse', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'purse'),
	(1103, '/var/appdata/models/classification/efficientnet/', 'wallet', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'wallet'),
	(1104, '/var/appdata/models/classification/efficientnet/', 'bib', 'C', 'I', 'efficientnet', 'imagenet', '3', '30', '172', 'bib'),
	(1105, '/var/appdata/models/classification/efficientnet/', 'parachute', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'parachute'),
	(1106, '/var/appdata/models/classification/efficientnet/', 'sleeping_bag', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'sleeping_bag'),
	(1107, '/var/appdata/models/classification/efficientnet/', 'television', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'television'),
	(1108, '/var/appdata/models/classification/efficientnet/', 'swimming_trunks', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'swimming_trunks'),
	(1109, '/var/appdata/models/classification/efficientnet/', 'measuring_cup', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'measuring_cup'),
	(1110, '/var/appdata/models/classification/efficientnet/', 'espresso', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'espresso');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1111, '/var/appdata/models/classification/efficientnet/', 'pizza', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'pizza'),
	(1112, '/var/appdata/models/classification/efficientnet/', 'breastplate', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'breastplate'),
	(1113, '/var/appdata/models/classification/efficientnet/', 'shopping_basket', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'shopping_basket'),
	(1114, '/var/appdata/models/classification/efficientnet/', 'wooden_spoon', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'wooden_spoon'),
	(1115, '/var/appdata/models/classification/efficientnet/', 'saltshaker', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'saltshaker'),
	(1116, '/var/appdata/models/classification/efficientnet/', 'chocolate_sauce', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'chocolate_sauce'),
	(1117, '/var/appdata/models/classification/efficientnet/', 'parachute', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'parachute'),
	(1118, '/var/appdata/models/classification/efficientnet/', 'ballplayer', 'C', 'I', 'efficientnet', 'imagenet', '1', '1', '172', 'ballplayer'),
	(1119, '/var/appdata/models/classification/efficientnet/', 'goblet', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'goblet'),
	(1120, '/var/appdata/models/classification/efficientnet/', 'gyromitra', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'gyromitra');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1121, '/var/appdata/models/classification/efficientnet/', 'dial_telephone', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'dial_telephone'),
	(1122, '/var/appdata/models/classification/efficientnet/', 'soap_dispenser', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'soap_dispenser'),
	(1123, '/var/appdata/models/classification/efficientnet/', 'jersey', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'jersey'),
	(1124, '/var/appdata/models/classification/efficientnet/', 'school_bus', 'C', 'I', 'efficientnet', 'imagenet', '3', '31', '172', 'school_bus'),
	(1125, '/var/appdata/models/classification/efficientnet/', 'jigsaw_puzzle', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'jigsaw_puzzle'),
	(1126, '/var/appdata/models/classification/efficientnet/', 'plastic_bag', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'plastic_bag'),
	(1127, '/var/appdata/models/classification/efficientnet/', 'reflex_camera', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'reflex_camera'),
	(1128, '/var/appdata/models/classification/efficientnet/', 'diaper', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'diaper'),
	(1129, '/var/appdata/models/classification/efficientnet/', 'Band_Aid', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'Band_Aid'),
	(1130, '/var/appdata/models/classification/efficientnet/', 'ice_lolly', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'ice_lolly');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1131, '/var/appdata/models/classification/efficientnet/', 'velvet', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'velvet'),
	(1132, '/var/appdata/models/classification/efficientnet/', 'tennis_ball', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'tennis_ball'),
	(1133, '/var/appdata/models/classification/efficientnet/', 'gasmask', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'gasmask'),
	(1134, '/var/appdata/models/classification/efficientnet/', 'doormat', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'doormat'),
	(1135, '/var/appdata/models/classification/efficientnet/', 'Loafer', 'C', 'I', 'efficientnet', 'imagenet', '3', '60', '172', 'Loafer'),
	(1136, '/var/appdata/models/classification/efficientnet/', 'ice_cream', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'ice_cream'),
	(1137, '/var/appdata/models/classification/efficientnet/', 'pretzel', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'pretzel'),
	(1138, '/var/appdata/models/classification/efficientnet/', 'quilt', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'quilt'),
	(1139, '/var/appdata/models/classification/efficientnet/', 'maillot', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'maillot'),
	(1140, '/var/appdata/models/classification/efficientnet/', 'tape_player', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'tape_player');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1141, '/var/appdata/models/classification/efficientnet/', 'clog', 'C', 'I', 'efficientnet', 'imagenet', '3', '60', '172', 'clog'),
	(1142, '/var/appdata/models/classification/efficientnet/', 'iPod', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'iPod'),
	(1143, '/var/appdata/models/classification/efficientnet/', 'bolete', 'C', 'I', 'efficientnet', 'imagenet', '5', '58', '172', 'bolete'),
	(1144, '/var/appdata/models/classification/efficientnet/', 'scuba_diver', 'C', 'I', 'efficientnet', 'imagenet', '1', '1', '172', 'scuba_diver'),
	(1145, '/var/appdata/models/classification/efficientnet/', 'pitcher', 'C', 'I', 'efficientnet', 'imagenet', '1', '1', '172', 'pitcher'),
	(1146, '/var/appdata/models/classification/efficientnet/', 'matchstick', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'matchstick'),
	(1147, '/var/appdata/models/classification/efficientnet/', 'bikini', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'bikini'),
	(1148, '/var/appdata/models/classification/efficientnet/', 'sock', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'sock'),
	(1149, '/var/appdata/models/classification/efficientnet/', 'CD_player', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'CD_player'),
	(1150, '/var/appdata/models/classification/efficientnet/', 'lens_cap', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'lens_cap');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1151, '/var/appdata/models/classification/efficientnet/', 'thatch', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'thatch'),
	(1152, '/var/appdata/models/classification/efficientnet/', 'vault', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'vault'),
	(1153, '/var/appdata/models/classification/efficientnet/', 'beaker', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'beaker'),
	(1154, '/var/appdata/models/classification/efficientnet/', 'bubble', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'bubble'),
	(1155, '/var/appdata/models/classification/efficientnet/', 'cheeseburger', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'cheeseburger'),
	(1156, '/var/appdata/models/classification/efficientnet/', 'parallel_bars', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'parallel_bars'),
	(1157, '/var/appdata/models/classification/efficientnet/', 'flagpole', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'flagpole'),
	(1158, '/var/appdata/models/classification/efficientnet/', 'coffee_mug', 'C', 'I', 'efficientnet', 'imagenet', '3', '50', '172', 'coffee_mug'),
	(1159, '/var/appdata/models/classification/efficientnet/', 'rubber_eraser', 'C', 'I', 'efficientnet', 'imagenet', '3', '7', '172', 'rubber_eraser'),
	(1160, '/var/appdata/models/classification/efficientnet/', 'stole', 'C', 'I', 'efficientnet', 'imagenet', '3', '59', '172', 'stole');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1161, '/var/appdata/models/classification/efficientnet/', 'carbonara', 'C', 'I', 'efficientnet', 'imagenet', '5', '7', '172', 'carbonara'),
	(1162, '/var/appdata/models/classification/efficientnet/', 'dumbbell', 'C', 'I', 'efficientnet', 'imagenet', '3', '32', '172', 'dumbbell'),
	(1163, '/var/appdata/models/classification/inception/', 'kit_fox', 'C', 'I', 'inception', 'imagenet', '2', '23', '172', 'kit_fox'),
	(1164, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'giraffe', 'D', 'I', 'YOLOV3', 'mscoco', '9', '85', '136', 'giraffe'),
	(1165, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'bench', 'D', 'I', 'YOLOV3', 'mscoco', '10', '63', '136', 'bench'),
	(1166, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'giraffe', 'D', 'I', 'tiny_yolo', 'mscoco', '9', '85', '136', 'giraffe'),
	(1167, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'bench', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '63', '136', 'bench'),
	(1168, '/var/appdata/models/detection/yolo/yolov3_checkpoints/', 'apple', 'D', 'I', 'YOLOV3', 'mscoco', '10', '12', '136', 'apple'),
	(1169, '/var/appdata/models/detection/yolo/yolov3tiny_checkpoints/', 'apple', 'D', 'I', 'tiny_yolo', 'mscoco', '10', '12', '136', 'apple'),
	(1170, '/var/appdata/models/detection/efficientdet', 'person', 'D', 'I', 'EFFICIENTDET', 'mscoco', '8', '8', '136', 'person');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1171, '/var/appdata/models/detection/efficientdet', 'bicycle', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'bicycle'),
	(1172, '/var/appdata/models/detection/efficientdet', 'car', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'car'),
	(1173, '/var/appdata/models/detection/efficientdet', 'motorbike', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'motorbike'),
	(1174, '/var/appdata/models/detection/efficientdet', 'aeroplane', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'aeroplane'),
	(1175, '/var/appdata/models/detection/efficientdet', 'bus', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'bus'),
	(1176, '/var/appdata/models/detection/efficientdet', 'train', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'train'),
	(1177, '/var/appdata/models/detection/efficientdet', 'truck', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'truck'),
	(1178, '/var/appdata/models/detection/efficientdet', 'boat', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '62', '136', 'boat'),
	(1179, '/var/appdata/models/detection/efficientdet', 'traffic light', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '63', '136', 'traffic light'),
	(1180, '/var/appdata/models/detection/efficientdet', 'fire hydrant', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '63', '136', 'fire hydrant');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1181, '/var/appdata/models/detection/efficientdet', 'stop sign', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '63', '136', 'stop sign'),
	(1182, '/var/appdata/models/detection/efficientdet', 'parking meter', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '63', '136', 'parking meter'),
	(1183, '/var/appdata/models/detection/efficientdet', 'bird', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '64', '136', 'bird'),
	(1184, '/var/appdata/models/detection/efficientdet', 'cat', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '65', '136', 'cat'),
	(1185, '/var/appdata/models/detection/efficientdet', 'dog', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '66', '136', 'dog'),
	(1186, '/var/appdata/models/detection/efficientdet', 'horse', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '67', '136', 'horse'),
	(1187, '/var/appdata/models/detection/efficientdet', 'sheep', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '68', '136', 'sheep'),
	(1188, '/var/appdata/models/detection/efficientdet', 'cow', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '69', '136', 'cow'),
	(1189, '/var/appdata/models/detection/efficientdet', 'elephant', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '70', '136', 'elephant'),
	(1190, '/var/appdata/models/detection/efficientdet', 'bear', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '71', '136', 'bear');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1191, '/var/appdata/models/detection/efficientdet', 'zebra', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '72', '136', 'zebra'),
	(1192, '/var/appdata/models/detection/efficientdet', 'backpack', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '73', '136', 'backpack'),
	(1193, '/var/appdata/models/detection/efficientdet', 'umbrella', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '73', '136', 'umbrella'),
	(1194, '/var/appdata/models/detection/efficientdet', 'handbag', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '73', '136', 'handbag'),
	(1195, '/var/appdata/models/detection/efficientdet', 'tie', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '73', '136', 'tie'),
	(1196, '/var/appdata/models/detection/efficientdet', 'suitcase', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '73', '136', 'suitcase'),
	(1197, '/var/appdata/models/detection/efficientdet', 'frisbee', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '74', '136', 'frisbee'),
	(1198, '/var/appdata/models/detection/efficientdet', 'skis', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '74', '136', 'skis'),
	(1199, '/var/appdata/models/detection/efficientdet', 'snowboard', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '74', '136', 'snowboard'),
	(1200, '/var/appdata/models/detection/efficientdet', 'sports ball', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '74', '136', 'sports ball');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1201, '/var/appdata/models/detection/efficientdet', 'kite', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '74', '136', 'kite'),
	(1202, '/var/appdata/models/detection/efficientdet', 'baseball bat', 'D', 'I', 'EFFICIENTDET', 'mscoco', '14', '74', '136', 'baseball bat'),
	(1203, '/var/appdata/models/detection/efficientdet', 'baseball glove', 'D', 'I', 'EFFICIENTDET', 'mscoco', '14', '74', '136', 'baseball glove'),
	(1204, '/var/appdata/models/detection/efficientdet', 'skateboard', 'D', 'I', 'EFFICIENTDET', 'mscoco', '14', '74', '136', 'skateboard'),
	(1205, '/var/appdata/models/detection/efficientdet', 'surfboard', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '74', '136', 'surfboard'),
	(1206, '/var/appdata/models/detection/efficientdet', 'tennis racket', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '74', '136', 'tennis racket'),
	(1207, '/var/appdata/models/detection/efficientdet', 'bottle', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '78', '136', 'bottle'),
	(1208, '/var/appdata/models/detection/efficientdet', 'wine glass', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '78', '136', 'wine glass'),
	(1209, '/var/appdata/models/detection/efficientdet', 'cup', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '78', '136', 'cup'),
	(1210, '/var/appdata/models/detection/efficientdet', 'fork', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '78', '136', 'fork');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1211, '/var/appdata/models/detection/efficientdet', 'knife', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '78', '136', 'knife'),
	(1212, '/var/appdata/models/detection/efficientdet', 'spoon', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '78', '136', 'spoon'),
	(1213, '/var/appdata/models/detection/efficientdet', 'bowl', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '78', '136', 'bowl'),
	(1214, '/var/appdata/models/detection/efficientdet', 'banana', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'banana'),
	(1215, '/var/appdata/models/detection/efficientdet', 'sandwich', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'sandwich'),
	(1216, '/var/appdata/models/detection/efficientdet', 'orange', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'orange'),
	(1217, '/var/appdata/models/detection/efficientdet', 'broccoli', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'broccoli'),
	(1218, '/var/appdata/models/detection/efficientdet', 'carrot', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'carrot'),
	(1219, '/var/appdata/models/detection/efficientdet', 'hot dog', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'hot dog'),
	(1220, '/var/appdata/models/detection/efficientdet', 'pizza', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'pizza');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1221, '/var/appdata/models/detection/efficientdet', 'donut', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'donut'),
	(1222, '/var/appdata/models/detection/efficientdet', 'cake', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'cake'),
	(1223, '/var/appdata/models/detection/efficientdet', 'chair', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '80', '136', 'chair'),
	(1224, '/var/appdata/models/detection/efficientdet', 'sofa', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '80', '136', 'sofa'),
	(1225, '/var/appdata/models/detection/efficientdet', 'pottedplant', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '81', '136', 'pottedplant'),
	(1226, '/var/appdata/models/detection/efficientdet', 'bed', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '80', '136', 'bed'),
	(1227, '/var/appdata/models/detection/efficientdet', 'diningtable', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '80', '136', 'diningtable'),
	(1228, '/var/appdata/models/detection/efficientdet', 'toilet', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '80', '136', 'toilet'),
	(1229, '/var/appdata/models/detection/efficientdet', 'tvmonitor', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'tvmonitor'),
	(1230, '/var/appdata/models/detection/efficientdet', 'laptop', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'laptop');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1231, '/var/appdata/models/detection/efficientdet', 'mouse', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'mouse'),
	(1232, '/var/appdata/models/detection/efficientdet', 'remote', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'remote'),
	(1233, '/var/appdata/models/detection/efficientdet', 'keyboard', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'keyboard'),
	(1234, '/var/appdata/models/detection/efficientdet', 'cell phone', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'cell phone'),
	(1235, '/var/appdata/models/detection/efficientdet', 'microwave', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'microwave'),
	(1236, '/var/appdata/models/detection/efficientdet', 'oven', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'oven'),
	(1237, '/var/appdata/models/detection/efficientdet', 'toaster', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'toaster'),
	(1238, '/var/appdata/models/detection/efficientdet', 'sink', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'sink'),
	(1239, '/var/appdata/models/detection/efficientdet', 'refrigerator', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '82', '136', 'refrigerator'),
	(1240, '/var/appdata/models/detection/efficientdet', 'book', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '84', '136', 'book');
INSERT INTO WEDA.PRE_TRAINED_CLASS
	(CLASS_CD,MDL_PATH,CLASS_DP_NAME,OBJECT_TYPE,MDL_TYPE,BASE_MDL,BASE_DATASET,CATEGORY1,CATEGORY2,CATEGORY3,CLASS_DB_NAME)
VALUES
	(1241, '/var/appdata/models/detection/efficientdet', 'clock', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '84', '136', 'clock'),
	(1242, '/var/appdata/models/detection/efficientdet', 'vase', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '84', '136', 'vase'),
	(1243, '/var/appdata/models/detection/efficientdet', 'scissors', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '84', '136', 'scissors'),
	(1244, '/var/appdata/models/detection/efficientdet', 'teddy bear', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '84', '136', 'teddy bear'),
	(1245, '/var/appdata/models/detection/efficientdet', 'hair drier', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '84', '136', 'hair drier'),
	(1246, '/var/appdata/models/detection/efficientdet', 'toothbrush', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '84', '136', 'toothbrush'),
	(1247, '/var/appdata/models/detection/efficientdet', 'giraffe', 'D', 'I', 'EFFICIENTDET', 'mscoco', '9', '85', '136', 'giraffe'),
	(1248, '/var/appdata/models/detection/efficientdet', 'bench', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '63', '136', 'bench'),
	(1249, '/var/appdata/models/detection/efficientdet', 'apple', 'D', 'I', 'EFFICIENTDET', 'mscoco', '10', '12', '136', 'apple');

INSERT INTO WEDA.USERS
	(USER_ID, USER_NM, USER_PW, `ROLE`, PREV_PW, `USE`, LOGIN_CNT, CRN_DTM, UPT_DTM) 
VALUES
  ('user', 'user', 'ddcd04e95616eb9c738e415a113af76f170bba5fccb3dc83d5dae68531681929544c7645aa9546188017245597ac6723435e48e414b8bbb6aad6bf647d43f165', 'ADMIN' , 'ddcd04e95616eb9c738e415a113af76f170bba5fccb3dc83d5dae68531681929544c7645aa9546188017245597ac6723435e48e414b8bbb6aad6bf647d43f165', '0' , '0' , 
  now(), now());

INSERT INTO LEGACY_MODELS
	(MDL_KIND,MDL_NM,OBJECT_TYPE,DATA_TYPE,MDL_PATH)
VALUES
	('ML', 'XGBClassifier', 'C', 'T', 'Network/Tabular/XGBOOST/XGB_CLF'),
	('ML', 'XGBRegressor', 'R', 'T', 'Network/Tabular/XGBOOST/XGB_REG'),
	('ML', 'RandomForestClassifier', 'C', 'T', 'Network/Tabular/SCIKIT/RF_CLF'),
	('ML', 'RandomForestRegressor', 'R', 'T', 'Network/Tabular/SCIKIT/RF_REG'),
	('ML', 'LGBMClassifier', 'C', 'T', 'Network/Tabular/LGBM/LGBM_CLF'),
	('ML', 'LGBMRegressor', 'R', 'T', 'Network/Tabular/LGBM/LGBM_REG'),
	('ML', 'SVC', 'C', 'T', 'Network/Tabular/SCIKIT/SVC'),
	('ML', 'SVR', 'R', 'T', 'Network/Tabular/SCIKIT/SVR'),
	('ML', 'LinearSVC', 'C', 'T', 'Network/Tabular/SCIKIT/LinearSVC'),
	('ML', 'LinearSVR', 'R', 'T', 'Network/Tabular/SCIKIT/LinearSVR'),
	('ML', 'CatboostClassifier', 'C', 'T', 'Network/Tabular/CATBOOST/CAT_CLF'),
	('ML', 'CatboostRegressor', 'R', 'T', 'Network/Tabular/CATBOOST/CAT_REG'),
	('ML', 'ExtraTreesClassifier', 'C', 'T', 'Network/Tabular/SCIKIT/ET_CLF'),
	('ML', 'ExtraTreesRegressor', 'R', 'T', 'Network/Tabular/SCIKIT/ET_REG'),
	('ML', 'HistGradientBoostingClassifier', 'C', 'T', 'Network/Tabular/SCIKIT/HIST_CLF'),
	('ML', 'HistGradientBoostingRegressor', 'R', 'T', 'Network/Tabular/SCIKIT/HIST_REG'),
	('ML', 'KNeighborsClassifier', 'C', 'T', 'Network/Tabular/SCIKIT/kNN_CLF'),
	('ML', 'KNeighborsRegressor', 'R', 'T', 'Network/Tabular/SCIKIT/kNN_REG'),
	('DL', 'TabNetCLF', 'C', 'T', 'Network/Tabular/TF/TabNetCLF'),
	('DL', 'DeepNeuralDecisionForest', 'C', 'T', 'Network/Tabular/PYTORCH/NEURAL_DECISION_FOREST'),
	('DL', 'AUTOKERASClassifier', 'C', 'T', 'Network/Tabular/AUTOKERAS/AUTOKERAS_CLF'),
	('DL', 'AUTOKERASRegressor', 'R', 'T', 'Network/Tabular/AUTOKERAS/AUTOKERAS_REG'),
	('FE', 'FEATURE_EXTRACTION', 'F', 'T', 'Network/Tabular/FEATURE_ENGINEERING/FEATURE_EXTRACTION/PCA'),
	('FE', 'FEATURE_SELECTION', 'F', 'T', 'Network/Tabular/FEATURE_ENGINEERING/FEATURE_SELECTION'),
	('ML', 'AUTOMLClassifier', 'C', 'T', 'Network/Tabular/AUTOML/AUTOML_CLF'),
	('ML', 'AUTOMLRegressor', 'R', 'T', 'Network/Tabular/AUTOML/AUTOML_REG');

INSERT INTO LEGACY_MODELOPTIONS
	(MDL_KIND,MDL_NM,OBJECT_TYPE,DATA_TYPE,PARAM)
VALUES
	("ML", "XGBClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"INT","DEFAULT_VALUE":6,"RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_child_weight","TYPE":"INT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"gamma","TYPE":"FLOAT","DEFAULT_VALUE":0,"RANGE":{"MIN":0,"MAX":"Infinity"}},{"PARAMETER_NAME":"colsample_bytree","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"colsample_bylevel","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"colsample_bynode","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"subsample","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.3,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "XGBRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"INT","DEFAULT_VALUE":6,"RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_child_weight","TYPE":"INT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"gamma","TYPE":"FLOAT","DEFAULT_VALUE":0,"RANGE":{"MIN":0,"MAX":"Infinity"}},{"PARAMETER_NAME":"colsample_bytree","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"colsample_bylevel","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"colsample_bynode","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"subsample","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.3,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "RandomForestClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_samples_split","TYPE":"INT","DEFAULT_VALUE":2,"RANGE":{"MIN":2,"MAX":"Infinity"}},{"PARAMETER_NAME":"min_samples_leaf","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"max_features","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"sqrt","value":"sqrt"},{"label":"log2","value":"log2"}]},{"PARAMETER_NAME":"max_leaf_nodes","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":1,"MAX":"Infinity"}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "RandomForestRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"max_samples_split","TYPE":"INT","DEFAULT_VALUE":2,"RANGE":{"MIN":2,"MAX":"Infinity"}},{"PARAMETER_NAME":"min_samples_leaf","TYPE":"INT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"max_features","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"sqrt","value":"sqrt"},{"label":"log2","value":"log2"}]},{"PARAMETER_NAME":"max_leaf_nodes","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":1,"MAX":"Infinity"}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "LGBMClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"INT","DEFAULT_VALUE":6,"RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"num_leaves","TYPE":"FLOAT","DEFAULT_VALUE":31,"RANGE":{"MIN":1,"MAX":131072}},{"PARAMETER_NAME":"colsample_bytree","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"colsample_bynode","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"subsample","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.1,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "LGBMRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"INT","DEFAULT_VALUE":6,"RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"num_leaves","TYPE":"INT","DEFAULT_VALUE":31,"RANGE":{"MIN":1,"MAX":131072}},{"PARAMETER_NAME":"colsample_bytree","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"colsample_bynode","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"subsample","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":1}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.1,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "SVC", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"max_iter","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"C","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":100}},{"PARAMETER_NAME":"kernel","TYPE":"STRING","DEFAULT_VALUE":"rbf","RANGE":[{"label":"poly","value":"poly"},{"label":"rbf","value":"rbf"},{"label":"sigmoid","value":"sigmoid"},{"label":"precomputed","value":"precomputed"}]},{"PARAMETER_NAME":"gamma","TYPE":"STRING","DEFAULT_VALUE":"scale","RANGE":[{"label":"scale","value":"scale"},{"label":"auto","value":"auto"}]},{"PARAMETER_NAME":"degree","TYPE":"INT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":10}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "SVR", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"max_iter","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"C","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":100}},{"PARAMETER_NAME":"kernel","TYPE":"STRING","DEFAULT_VALUE":"rbf","RANGE":[{"label":"poly","value":"poly"},{"label":"rbf","value":"rbf"},{"label":"sigmoid","value":"sigmoid"},{"label":"precomputed","value":"precomputed"}]},{"PARAMETER_NAME":"gamma","TYPE":"STRING","DEFAULT_VALUE":"scale","RANGE":[{"label":"scale","value":"scale"},{"label":"auto","value":"auto"}]},{"PARAMETER_NAME":"degree","TYPE":"INT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":10}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "LinearSVC", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"max_iter","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"C","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":100}},{"PARAMETER_NAME":"dual","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"loss","TYPE":"STRING","DEFAULT_VALUE":"squared_hinge","RANGE":[{"label":"squared_hinge","value":"squared_hinge"},{"label":"hinge","value":"hinge"}]}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "LinearSVR", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"max_iter","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"C","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":100}},{"PARAMETER_NAME":"dual","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"loss","TYPE":"STRING","DEFAULT_VALUE":"epsilon_insensitive","RANGE":[{"label":"epsilon_insensitive","value":"epsilon_insensitive"},{"label":"squared_epsilon_insensitive","value":"squared_epsilon_insensitive"}]}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "CatboostClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"iterations","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"depth","TYPE":"INT","DEFAULT_VALUE":6,"RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_data_in_leaf","TYPE":"INT","DEFAULT_VALUE":2,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"bagging_temperature","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":"Infinity"}},{"PARAMETER_NAME":"l2_leaf_reg","TYPE":"FLOAT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":30}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.5,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "CatboostRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"iterations","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"depth","TYPE":"INT","DEFAULT_VALUE":6,"RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_data_in_leaf","TYPE":"INT","DEFAULT_VALUE":2,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"bagging_temperature","TYPE":"FLOAT","DEFAULT_VALUE":1,"RANGE":{"MIN":0,"MAX":"Infinity"}},{"PARAMETER_NAME":"l2_leaf_reg","TYPE":"FLOAT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":30}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.5,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "ExtraTreesClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_samples_split","TYPE":"INT","DEFAULT_VALUE":2,"RANGE":{"MIN":2,"MAX":"Infinity"}},{"PARAMETER_NAME":"min_samples_leaf","TYPE":"INT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"max_features","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"sqrt","value":"sqrt"},{"label":"log2","value":"log2"}]},{"PARAMETER_NAME":"max_leaf_nodes","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":1,"MAX":"Infinity"}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "ExtraTreesRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_estimators","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_samples_split","TYPE":"INT","DEFAULT_VALUE":2,"RANGE":{"MIN":2,"MAX":"Infinity"}},{"PARAMETER_NAME":"min_samples_leaf","TYPE":"INT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"max_features","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"sqrt","value":"sqrt"},{"label":"log2","value":"log2"}]},{"PARAMETER_NAME":"max_leaf_nodes","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":1,"MAX":"Infinity"}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "HistGradientBoostingClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"max_iter","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_samples_leaf","TYPE":"INT","DEFAULT_VALUE":20,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"max_bins","TYPE":"INT","DEFAULT_VALUE":255,"RANGE":{"MIN":1,"MAX":255}},{"PARAMETER_NAME":"max_leaf_nodes","TYPE":"INT","DEFAULT_VALUE":31,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.1,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "HistGradientBoostingRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"max_iter","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"max_depth","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"min_samples_leaf","TYPE":"INT","DEFAULT_VALUE":20,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"max_bins","TYPE":"INT","DEFAULT_VALUE":255,"RANGE":{"MIN":1,"MAX":255}},{"PARAMETER_NAME":"max_leaf_nodes","TYPE":"INT","DEFAULT_VALUE":31,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.1,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "KNeighborsClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_neighbors","TYPE":"INT","DEFAULT_VALUE":5,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"weights","TYPE":"STRING","DEFAULT_VALUE":"uniform","RANGE":[{"label":"uniform","value":"uniform"},{"label":"distance","value":"distance"}]},{"PARAMETER_NAME":"leaf_size","TYPE":"INT","DEFAULT_VALUE":30,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"algolithm","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"ball_tree","value":"ball_tree"},{"label":"kd_tree","value":"kd_tree"},{"label":"brute","value":"brute"}]}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "KNeighborsRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"n_neighbors","TYPE":"INT","DEFAULT_VALUE":5,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"weights","TYPE":"STRING","DEFAULT_VALUE":"uniform","RANGE":[{"label":"uniform","value":"uniform"},{"label":"distance","value":"distance"}]},{"PARAMETER_NAME":"leaf_size","TYPE":"INT","DEFAULT_VALUE":30,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"algolithm","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"ball_tree","value":"ball_tree"},{"label":"kd_tree","value":"kd_tree"},{"label":"brute","value":"brute"}]}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("DL", "TabNetCLF", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"epochs","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"batch_size","TYPE":"INT","DEFAULT_VALUE":32,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"num_decision_step","TYPE":"INT","DEFAULT_VALUE":5,"RANGE":{"MIN":3,"MAX":10}},{"PARAMETER_NAME":"relaxation_factor","TYPE":"FLOAT","DEFAULT_VALUE":0.5,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"sparsity_coefficient","TYPE":"FLOAT","DEFAULT_VALUE":0.00005,"RANGE":{"MIN":0,"MAX":0.00005}},{"PARAMETER_NAME":"batch_momentum","TYPE":"FLOAT","DEFAULT_VALUE":0.98,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"virtual_batch_size","TYPE":"STR/INT","DEFAULT_VALUE":"None","RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"num_groups","TYPE":"INT","DEFAULT_VALUE":1,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.01,"RANGE":{"MIN":0,"MAX":1}}],"OPTIMIZER":[{"PARAMETER_NAME":"optimizer","TYPE":"STRING","DEFAULT_VALUE":"adam","RANGE":[{"label":"adadelta","value":"adadelta"},{"label":"adagrad","value":"adagrad"},{"label":"adam","value":"adam"},{"label":"adamax","value":"adamax"},{"label":"nadam","value":"nadam"},{"label":"rmsprop","value":"rmsprop"},{"label":"sgd","value":"sgd"}]}],"LOSS":[{"PARAMETER_NAME":"loss","TYPE":"STRING","DEFAULT_VALUE":"categorical_crossentropy","RANGE":[{"label":"categorical_crossentropy","value":"categorical_crossentropy"},{"label":"binary_crossentropy","value":"binary_crossentropy"},{"label":"sparse_categoricalcrossentropy","value":"sparse_categoricalcrossentropy"}]}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"loss","value":"loss"},{"label":"val_loss","value":"val_loss"},{"label":"accuracy","value":"accuracy"},{"label":"val_accuracy","value":"val_accuracy"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("DL", "DeepNeuralDecisionForest", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"epochs","TYPE":"INT","DEFAULT_VALUE":100,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"DEPTH":[{"PARAMETER_NAME":"tree_depth","TYPE":"INT","DEFAULT_VALUE":3,"RANGE":{"MIN":3,"MAX":10}}],"SAMPLING":[{"PARAMETER_NAME":"batch_size","TYPE":"INT","DEFAULT_VALUE":128,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"n_tree","TYPE":"INT","DEFAULT_VALUE":5,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"tree_feature_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.5,"RANGE":{"MIN":0,"MAX":1}},{"PARAMETER_NAME":"feat_dropout","TYPE":"FLOAT","DEFAULT_VALUE":0.3,"RANGE":{"MIN":0,"MAX":1}}],"LEARNING_RATE":[{"PARAMETER_NAME":"learning_rate","TYPE":"FLOAT","DEFAULT_VALUE":0.003,"RANGE":{"MIN":0,"MAX":1}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"loss","value":"loss"},{"label":"accuracy","value":"accuracy"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("DL", "AUTOKERASClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"epochs","TYPE":"INT","DEFAULT_VALUE":30,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"batch_size","TYPE":"INT","DEFAULT_VALUE":32,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"max_trials","TYPE":"INT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"tunner","TYPE":"STRING","DEFAULT_VALUE":"greedy","RANGE":[{"label":"greedy","value":"greedy"},{"label":"bayesian","value":"bayesian"},{"label":"hyperband","value":"hyperband"},{"label":"random","value":"random"}]}],"LOSS":[{"PARAMETER_NAME":"loss","TYPE":"STRING","DEFAULT_VALUE":"categorical_crossentropy","RANGE":[{"label":"categorical_crossentropy","value":"categorical_crossentropy"},{"label":"binary_crossentropy","value":"binary_crossentropy"},{"label":"sparse_categoricalcrossentropy","value":"sparse_categoricalcrossentropy"}]}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"loss","value":"loss"},{"label":"val_loss","value":"val_loss"},{"label":"accuracy","value":"accuracy"},{"label":"val_accuracy","value":"val_accuracy"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("DL", "AUTOKERASRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"epochs","TYPE":"INT","DEFAULT_VALUE":30,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"SAMPLING":[{"PARAMETER_NAME":"batch_size","TYPE":"INT","DEFAULT_VALUE":32,"RANGE":{"MIN":1,"MAX":"Infinity"}},{"PARAMETER_NAME":"max_trials","TYPE":"INT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":10}},{"PARAMETER_NAME":"tunner","TYPE":"STRING","DEFAULT_VALUE":"greedy","RANGE":[{"label":"greedy","value":"greedy"},{"label":"bayesian","value":"bayesian"},{"label":"hyperband","value":"hyperband"},{"label":"random","value":"random"}]}],"LOSS":[{"PARAMETER_NAME":"loss","TYPE":"STRING","DEFAULT_VALUE":"mean_squared_error","RANGE":[{"label":"mean_squared_error","value":"mean_squared_error"},{"label":"mean_absolute_error","value":"mean_absolute_error"},{"label":"mean_absolute_percentage_error","value":"mean_absolute_percentage_error"},{"label":"mean_squared_logarithmic_error","value":"mean_squared_logarithmic_error"},{"label":"huber_loss","value":"huber_loss"},{"label":"log_cosh","value":"log_cosh"}]}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"loss","RANGE":[{"label":"loss","value":"loss"},{"label":"val_loss","value":"val_loss"},{"label":"mean_squared_error","value":"mean_squared_error"},{"label":"val_mean_squared_error","value":"val_mean_squared_error"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("FE", "FEATURE_EXTRACTION", "F", "T", '{}'),
	("FE", "FEATURE_SELECTION", "F", "T", '{}'),
	("ML", "AUTOMLClassifier", "C", "T", '{"EPOCH":[{"PARAMETER_NAME":"epochs","TYPE":"INT","DEFAULT_VALUE":20,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"ALGORITHM":[{"PARAMETER_NAME":"algorithm","TYPE":"STRING","DEFAULT_VALUE":"random","RANGE":[{"label":"greedy","value":"greedy"},{"label":"random","value":"random"},{"label":"bayesian","value":"bayesian"}]}],"DEPTH":[{"PARAMETER_NAME":"max_trial","TYPE":"INT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"accuracy","RANGE":[{"label":"accuracy","value":"accuracy"},{"label":"precision","value":"precision"},{"label":"recall","value":"recall"},{"label":"f1","value":"f1"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}'),
	("ML", "AUTOMLRegressor", "R", "T", '{"EPOCH":[{"PARAMETER_NAME":"epochs","TYPE":"INT","DEFAULT_VALUE":20,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"ALGORITHM":[{"PARAMETER_NAME":"algorithm","TYPE":"STRING","DEFAULT_VALUE":"random","RANGE":[{"label":"greedy","value":"greedy"},{"label":"random","value":"random"},{"label":"bayesian","value":"bayesian"}]}],"DEPTH":[{"PARAMETER_NAME":"max_trial","TYPE":"INT","DEFAULT_VALUE":3,"RANGE":{"MIN":1,"MAX":"Infinity"}}],"EARLYSTOPPING":[{"PARAMETER_NAME":"early_stopping","TYPE":"BOOL","DEFAULT_VALUE":"TRUE","RANGE":[{"label":"TRUE","value":"TRUE"},{"label":"FALSE","value":"FALSE"}]},{"PARAMETER_NAME":"monitor","TYPE":"STRING","DEFAULT_VALUE":"r2","RANGE":[{"label":"r2","value":"r2"},{"label":"mae","value":"mae"},{"label":"mse","value":"mse"},{"label":"rmse","value":"rmse"}]},{"PARAMETER_NAME":"mode","TYPE":"STRING","DEFAULT_VALUE":"auto","RANGE":[{"label":"auto","value":"auto"},{"label":"min","value":"min"},{"label":"max","value":"max"}]}]}');