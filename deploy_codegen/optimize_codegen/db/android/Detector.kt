package com.surendramaran.yolov9tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.surendramaran.yolov9tflite.MetaData.extractNamesFromLabelFile
import com.surendramaran.yolov9tflite.MetaData.extractNamesFromMetadata
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String?,
    private val detectorListener: DetectorListener,
    private val message: (String) -> Unit
) {

    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        val compatList = CompatibilityList()

        val options = Interpreter.Options().apply{
            if(compatList.isDelegateSupportedOnThisDevice){
                val delegateOptions = compatList.bestOptionsForThisDevice
                this.addDelegate(GpuDelegate(delegateOptions))
            } else {
                this.setNumThreads(4)
            }
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        labels.addAll(extractNamesFromMetadata(model))
        if (labels.isEmpty()) {
            if (labelPath == null) {
                message("Model not contains metadata, provide LABELS_PATH in Constants.kt")
                labels.addAll(MetaData.TEMP_CLASSES)
            } else {
                labels.addAll(extractNamesFromLabelFile(context, labelPath))
            }
        }

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numChannel = outputShape[1]
            numElements = outputShape[2]
        }
    }

    fun restart(delegate: Delegate){
        interpreter.close()

        var options = Interpreter.Options()

        when(delegate){
            Delegate.CPU -> {
                options = options.apply { this.setNumThreads(4) }
            }
            Delegate.GPU -> {
                val compatList = CompatibilityList()
                options = options.apply {
                    if (compatList.isDelegateSupportedOnThisDevice) {
                        val delegateOptions = compatList.bestOptionsForThisDevice
                        this.addDelegate(GpuDelegate(delegateOptions))
                    } else {
                        this.setNumThreads(4)
                    }
                }
            }
            Delegate.NNAPI -> {
                options = options.apply {
                    this.addDelegate(NnApiDelegate())
                    this.setUseNNAPI(true)
                }
            }
        }
        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)
    }

    fun restart(isGpu: Boolean) {
        interpreter.close()

        val options = if (isGpu) {
            val compatList = CompatibilityList()
            Interpreter.Options().apply{
                if(compatList.isDelegateSupportedOnThisDevice){
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    this.addDelegate(GpuDelegate(delegateOptions))
                } else {
                    this.setNumThreads(4)
                }
            }
        } else {
            Interpreter.Options().apply{
                this.setNumThreads(4)
            }
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)
    }

    fun close() {
        interpreter.close()
    }

    fun detect(frame: Bitmap) {
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

//        Log.i("WIDTH", tensorWidth.toString())
//        Log.i("HEIGHT", tensorHeight.toString())
        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect(inferenceTime)
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? {

        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]
                val cx = (array[c] / tensorWidth) // 0
                val cy = (array[c + numElements] /tensorWidth) // 1
                val w = (array[c + numElements * 2] / tensorWidth)
                val h = (array[c + numElements * 3] / tensorWidth)
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface DetectorListener {
        fun onEmptyDetect(inferenceTime: Long)
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }
}
