package com.oddysw.detectorocv

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraActivity
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.IOException
import java.io.InputStream


class MainActivity : CameraActivity(), CvCameraViewListener2 {

    private fun isCameraPermissionGranted(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {

        when (requestCode) {
            CAMERA_PERMISSION_CODE -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Camera access granted", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Camera access denied", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    public override fun onResume() {
        super.onResume()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.enableView()
        if (!isCameraPermissionGranted()) requestCameraPermission()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!isCameraPermissionGranted()) requestCameraPermission()
        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully")
        } else {
            Log.e(TAG, "OpenCV initialization failed!")
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show()
            return
        }
        mModelBuffer = loadFileFromResource("mobilenet_iter_73000.caffemodel")
        mConfigBuffer = loadFileFromResource("deploy.prototxt")
        if (mModelBuffer == null || mConfigBuffer == null) {
            Log.e(TAG, "Failed to load model from resources")
        } else Log.i(TAG, "Model files loaded successfully")
        net = Dnn.readNet("caffe", mModelBuffer, mConfigBuffer)
        Log.i(TAG, "Network loaded successfully")
        setContentView(R.layout.activity_main)

        // Set up camera listener.

        mOpenCvCameraView = findViewById<View>(R.id.CameraView) as CameraBridgeViewBase
        mOpenCvCameraView!!.visibility = CameraBridgeViewBase.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)
    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    override fun getCameraViewList(): List<CameraBridgeViewBase?> {
        return listOf(mOpenCvCameraView)
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
        mModelBuffer!!.release()
        mConfigBuffer!!.release()
    }

    // Load a network.
    override fun onCameraViewStarted(width: Int, height: Int) {}
    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        val IN_WIDTH = 300
        val IN_HEIGHT = 300
        val WH_RATIO = IN_WIDTH.toFloat() / IN_HEIGHT
        val IN_SCALE_FACTOR = 0.007843
        val MEAN_VAL = 127.5
        val THRESHOLD = 0.2

        // Get a new frame
        Log.d(TAG, "handle new frame!")
        val frame = inputFrame.rgba()
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)

        // Forward image through network.
        val blob = Dnn.blobFromImage(
            frame, IN_SCALE_FACTOR,
            Size(IN_WIDTH.toDouble(), IN_HEIGHT.toDouble()),
            Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL),  /*swapRB*/false,  /*crop*/false
        )
        net!!.setInput(blob)
        var detections = net!!.forward()
        val cols = frame.cols()
        val rows = frame.rows()
        detections = detections.reshape(1, detections.total().toInt() / 7)
        for (i in 0 until detections.rows()) {
            val confidence = detections[i, 2][0]
            if (confidence > THRESHOLD) {
                val classId = detections[i, 1][0].toInt()
                val left = (detections[i, 3][0] * cols).toInt()
                val top = (detections[i, 4][0] * rows).toInt()
                val right = (detections[i, 5][0] * cols).toInt()
                val bottom = (detections[i, 6][0] * rows).toInt()

                // Draw rectangle around detected object.
                Imgproc.rectangle(
                    frame,
                    Point(left.toDouble(), top.toDouble()),
                    Point(right.toDouble(), bottom.toDouble()),
                    Scalar(0.0, 255.0, 0.0)
                )
                val label = classNames[classId] + ": " + confidence
                val baseLine = IntArray(1)
                val labelSize =
                    Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine)

                // Draw background for label.
                Imgproc.rectangle(
                    frame, Point(left.toDouble(), top - labelSize.height),
                    Point(left + labelSize.width, (top + baseLine[0]).toDouble()),
                    Scalar(255.0, 255.0, 255.0), Imgproc.FILLED
                )
                // Write class name and confidence.
                Imgproc.putText(
                    frame, label, Point(left.toDouble(), top.toDouble()),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0.0, 0.0, 0.0)
                )
            }
        }
        return frame
    }

    override fun onCameraViewStopped() {}
    private fun loadFileFromResource(filePath: String): MatOfByte? {
        val errorString: String = "Failed to load model from resources!"
        return try {
                val assetManager: AssetManager = applicationContext.assets
                val inputStream: InputStream = assetManager.open(filePath)
                val size = inputStream.available()
                val buffer = ByteArray(size)
                inputStream.read(buffer)
                inputStream.close()
                MatOfByte(*buffer)
            } catch (e: IOException) {
                e.printStackTrace()
                Log.e(
                    TAG,
                    "$errorString Exception thrown: $e"
                )
                Toast.makeText(this, errorString, Toast.LENGTH_LONG).show()
                null
            }

    }

    private var mConfigBuffer: MatOfByte? = null
    private var mModelBuffer: MatOfByte? = null
    private var net: Net? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null

    companion object {
        private const val CAMERA_PERMISSION_CODE = 100
        private const val TAG = "Detector_OpenCV"
        private val classNames = arrayOf(
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        )
    }
}
