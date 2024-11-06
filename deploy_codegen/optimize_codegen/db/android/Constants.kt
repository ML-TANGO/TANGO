package com.surendramaran.yolov9tflite

object Constants {
    const val MODEL_PATH = "@@filename@@"  
    val LABELS_PATH: String = "labels.txt"
}

enum class Delegate{
    CPU, GPU, NNAPI
}
