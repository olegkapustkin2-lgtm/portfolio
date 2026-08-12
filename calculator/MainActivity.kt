package com.example.calculator_pet

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.MotionEvent
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.calculator_pet.ui.theme.Calculator_petTheme
import net.objecthunter.exp4j.ExpressionBuilder
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.hypot
import org.json.JSONObject
//import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
class MainActivity : ComponentActivity() {
    private var model: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(arrayOf(
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE
            ), 100)
        }

        try {
            model = Interpreter(loadModelFile("calculator_model.tflite"))
            Toast.makeText(this, "✅ Модель загружена", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "❌ Ошибка загрузки модели: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
        }

        setContent {
            Calculator_petTheme {
                DrawingApp(model)
            }
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val assetFileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        model?.close()
        super.onDestroy()
    }
}

// ====== UI ======
@Composable
fun DrawingApp(model: Interpreter?) {
    var mode by remember { mutableStateOf("calculator") }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = { mode = "calculator" },
                modifier = Modifier.weight(1f),
                colors = if (mode == "calculator") ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                else ButtonDefaults.buttonColors()
            ) {
                Text("🧮 Калькулятор")
            }
            Button(
                onClick = { mode = "collector" },
                modifier = Modifier.weight(1f),
                colors = if (mode == "collector") ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                else ButtonDefaults.buttonColors()
            ) {
                Text("📊 Сбор данных")
            }
        }

        when (mode) {
            "calculator" -> CalculatorMode()
//            "collector" -> DataCollectorMode()
        }
    }
}

// ====== КАЛЬКУЛЯТОР ======
@Composable
fun CalculatorMode() {
    val completedStrokes = remember { mutableStateListOf<List<Offset>>() }
    var currentStrokePoints by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var recognitionResult by remember { mutableStateOf("") }
    var history by remember { mutableStateOf(listOf<String>()) }
    val context = LocalContext.current
    var canvasSize by remember { mutableStateOf(Size.Zero) }
    val classifier = remember { SymbolClassifier(context) }

    Column(modifier = Modifier.fillMaxSize()) {
        // Результат
        Text(
            text = if (recognitionResult.isEmpty()) "Напишите пример" else recognitionResult,
            modifier = Modifier.padding(16.dp)
        )

        // История
//        if (history.isNotEmpty()) {
//            Column(modifier = Modifier.padding(horizontal = 16.dp)) {
//                Text("📜 История:", style = MaterialTheme.typography.labelMedium)
//                history.takeLast(5).forEach { entry ->
//                    Text("  $entry", style = MaterialTheme.typography.bodySmall)
//                }
//            }
//        }



        Box(modifier = Modifier.fillMaxWidth().weight(1f)) {
            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInteropFilter { event ->
                        when (event.action) {
                            MotionEvent.ACTION_DOWN -> {
                                currentStrokePoints = listOf(Offset(event.x, event.y))
                                true
                            }
                            MotionEvent.ACTION_MOVE -> {
                                val newPoint = Offset(event.x, event.y)
                                if (currentStrokePoints.isNotEmpty()) {
                                    val lastPoint = currentStrokePoints.last()
                                    val distance = hypot(newPoint.x - lastPoint.x, newPoint.y - lastPoint.y)
                                    if (distance > 3f) {
                                        currentStrokePoints = currentStrokePoints + newPoint
                                    }
                                } else {
                                    currentStrokePoints = listOf(newPoint)
                                }
                                true
                            }
                            MotionEvent.ACTION_UP -> {
                                if (currentStrokePoints.isNotEmpty()) {
                                    completedStrokes.add(currentStrokePoints)
                                }
                                currentStrokePoints = emptyList()
                                true
                            }
                            else -> false
                        }
                    }
            ) {
                // ✅ Рисуем все завершенные штрихи
                for (stroke in completedStrokes) {
                    drawStrokePath(stroke)
                }
                // ✅ Рисуем текущий штрих
                drawStrokePath(currentStrokePoints)

                // Сохраняем размер холста
                canvasSize = this.size
            }
        }





        // Кнопки
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Button(
                onClick = {
                    completedStrokes.clear()
                    currentStrokePoints = emptyList()
                    recognitionResult = ""
                },
                modifier = Modifier.weight(1f)
            ) {
                Text("🗑️ Очистить")
            }

            Button(
                onClick = {
                    if (completedStrokes.isNotEmpty()) {
                        completedStrokes.removeAt(completedStrokes.size - 1)
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Text("↩️ Отмена")
            }

            Button(
                onClick = {
                    if (completedStrokes.isEmpty() && currentStrokePoints.isEmpty()) {
                        Toast.makeText(context, "Сначала нарисуйте выражение", Toast.LENGTH_SHORT).show()
                        return@Button
                    }

                    val allStrokes = completedStrokes + if (currentStrokePoints.isNotEmpty()) listOf(currentStrokePoints) else emptyList()
                    val bitmap = createBitmapFromStrokes(allStrokes, context, canvasSize)

                    if (bitmap == null) {
                        recognitionResult = "Ошибка создания изображения"
                        return@Button
                    }

                    val segmenter = SymbolSegmenter()
                    val symbols = segmenter.segmentSymbols(bitmap)

                    if (symbols.isEmpty()) {
                        recognitionResult = "Символы не найдены"
                        return@Button
                    }

                    val recognized = symbols.mapNotNull {
                        val centered = centerAndResize(it, 64)
                        classifier.classifySymbol(centered)
                    }

                    if (recognized.isEmpty()) {
                        recognitionResult = "Не удалось распознать"
                        return@Button
                    }

                    val rawExpression = recognized.joinToString("")

                    val exprForCalc = rawExpression
                        .replace("√", "sqrt")
                        .replace("×", "*")
                        .replace("÷", "/")
                        .replace("|", "abs")
                        .replace("=", "=")

                    val result = try {
                        ExpressionBuilder(exprForCalc).build().evaluate()
                    } catch (e: Exception) {
                        null
                    }

                    recognitionResult = if (result != null) {
                        val entry = "$rawExpression = $result"
                        history = history + entry
                        entry
                    } else {
                        rawExpression
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Text("🧮 Вычислить")
            }
        }
    }
}

// ====== СБОР ДАННЫХ ======
//@Composable
//fun DataCollectorMode(startFrom: String = "0") {
//    val symbols = listOf(
//        "0","1","2","3","4","5","6","7","8","9",
//        "+","-","×","÷",
//        "√","|","=","(",")","?",
//        ","
//    )
//
//    val startIndex = symbols.indexOf(startFrom).takeIf { it >= 0 } ?: 0
//    var currentIndex by remember { mutableStateOf(startIndex) }
//    var counter by remember { mutableStateOf(0) }
//    var completedStrokes by remember { mutableStateOf<List<List<Offset>>>(emptyList()) }
//    var currentStroke by remember { mutableStateOf<List<Offset>>(emptyList()) }
//    val context = LocalContext.current
//    val targetCount = 100
//
//    val progress = (counter.toFloat() / targetCount * 100).toInt()
//    val totalProgress = ((currentIndex * targetCount + counter).toFloat() / (symbols.size * targetCount) * 100).toInt()
//
//    Column(modifier = Modifier.fillMaxSize()) {
//        Card(
//            modifier = Modifier.fillMaxWidth().padding(8.dp),
//            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
//        ) {
//            Column(modifier = Modifier.padding(16.dp)) {
//                Text("Рисуй: ${symbols[currentIndex]}", style = MaterialTheme.typography.headlineSmall)
//                Text("Пример $counter/$targetCount (${progress}%)", style = MaterialTheme.typography.bodyMedium)
//                LinearProgressIndicator(progress = { counter.toFloat() / targetCount }, modifier = Modifier.fillMaxWidth())
//                Text("Всего: $totalProgress%", style = MaterialTheme.typography.bodySmall)
//            }
//        }
//
//        Box(modifier = Modifier.fillMaxWidth().weight(1f)) {
//            Canvas(
//                modifier = Modifier
//                    .fillMaxSize()
//                    .pointerInteropFilter { event ->
//                        when (event.action) {
//                            MotionEvent.ACTION_DOWN -> {
//                                currentStroke = listOf(Offset(event.x, event.y))
//                                true
//                            }
//                            MotionEvent.ACTION_MOVE -> {
//                                val newPoint = Offset(event.x, event.y)
//                                if (currentStroke.isNotEmpty()) {
//                                    val lastPoint = currentStroke.last()
//                                    val distance = hypot(newPoint.x - lastPoint.x, newPoint.y - lastPoint.y)
//                                    if (distance > 5f) {
//                                        currentStroke = currentStroke + newPoint
//                                    }
//                                } else {
//                                    currentStroke = listOf(newPoint)
//                                }
//                                true
//                            }
//                            MotionEvent.ACTION_UP -> {
//                                if (currentStroke.isNotEmpty()) {
//                                    completedStrokes = completedStrokes + listOf(currentStroke)
//                                }
//                                currentStroke = emptyList()
//                                true
//                            }
//                            else -> false
//                        }
//                    }
//            ) {
//                for (stroke in completedStrokes) drawLineStroke(stroke)
//                drawLineStroke(currentStroke)
//            }
//        }
//
//        Row(
//            modifier = Modifier.fillMaxWidth().padding(8.dp),
//            horizontalArrangement = Arrangement.spacedBy(8.dp)
//        ) {
//            Button(
//                onClick = {
//                    if (completedStrokes.isNotEmpty()) {
//                        val symbol = symbols[currentIndex]
//                        saveSymbol(completedStrokes, symbol, context)
//                        counter++
//                        completedStrokes = emptyList()
//                        currentStroke = emptyList()
//
//                        if (counter >= targetCount) {
//                            if (currentIndex < symbols.size - 1) {
//                                currentIndex++
//                                counter = 0
//                                Toast.makeText(context, "✅ Теперь рисуй: ${symbols[currentIndex]}", Toast.LENGTH_SHORT).show()
//                            } else {
//                                Toast.makeText(context, "🎉 Датасет собран! Всего ${symbols.size * targetCount} изображений.", Toast.LENGTH_LONG).show()
//                            }
//                        }
//                    } else {
//                        Toast.makeText(context, "Сначала нарисуйте символ", Toast.LENGTH_SHORT).show()
//                    }
//                },
//                modifier = Modifier.weight(1f)
//            ) {
//                Text("💾 Сохранить (${counter + 1}/$targetCount)")
//            }
//
//            Button(
//                onClick = {
//                    completedStrokes = emptyList()
//                    currentStroke = emptyList()
//                },
//                modifier = Modifier.weight(1f)
//            ) {
//                Text("🗑️ Очистить")
//            }
//        }
//    }
//}

// ====== КЛАССИФИКАТОР ======
class SymbolClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val classNames: List<String>
    private val inputSize = 64

    init {
        try {
            val model = loadModelFile("calculator_model.tflite")
            interpreter = Interpreter(model)
            Log.d("TFLite", "✅ Модель загружена")
        } catch (e: Exception) {
            Log.e("TFLite", "❌ Ошибка загрузки модели", e)
        }
        classNames = loadClassMapping()
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadClassMapping(): List<String> {
        return try {
            val json = context.assets.open("class_mapping_android.json").bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            val map = mutableMapOf<Int, String>()
            obj.keys().forEach { key -> map[key.toInt()] = obj.getString(key) }
            map.toSortedMap().values.toList()
        } catch (e: Exception) {
            Log.e("TFLite", "❌ Ошибка загрузки маппинга", e)
            listOf("?")
        }
    }

    fun classifySymbol(bitmap: Bitmap): String? {
        val interpreter = interpreter ?: return null
        if (bitmap.width < 10 || bitmap.height < 10) return "?"

        val input = preprocessBitmap(bitmap)
        val output = Array(1) { FloatArray(classNames.size) }

        try {
            interpreter.run(input, output)
        } catch (e: Exception) {
            Log.e("TFLite", "❌ Ошибка выполнения модели", e)
            return null
        }

        val predictions = output[0]
        val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: return null
        val confidence = predictions[maxIndex]

        return if (confidence > 0.3 && maxIndex < classNames.size) {
            classNames[maxIndex]
        } else {
            "?"
        }
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 1 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            val gray = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
            buffer.putFloat(gray.toFloat())
        }

        buffer.rewind()
        return buffer
    }
}

// ====== СЕГМЕНТАТОР ======
class SymbolSegmenter {
    private var bitmapWidth = 0
    private var bitmapHeight = 0

    fun segmentSymbols(bitmap: Bitmap): List<Bitmap> {
        bitmapWidth = bitmap.width
        bitmapHeight = bitmap.height

        val components = findConnectedComponents(bitmap)
        val filtered = components.filter { it.width() > 8 && it.height() > 8 }
        return filtered.sortedBy { it.left }.map { cropToBounds(bitmap, it) }
    }

    private fun findConnectedComponents(bitmap: Bitmap): List<Rect> {
        val visited = Array(bitmap.width) { BooleanArray(bitmap.height) }
        val components = mutableListOf<Rect>()

        for (x in 0 until bitmap.width) {
            for (y in 0 until bitmap.height) {
                if (!visited[x][y] && isPixelBlack(bitmap, x, y)) {
                    val points = mutableListOf<Pair<Int, Int>>()
                    floodFill(bitmap, x, y, visited, points)
                    components.add(getBounds(points))
                }
            }
        }
        return components
    }

    private fun floodFill(
        bitmap: Bitmap,
        startX: Int,
        startY: Int,
        visited: Array<BooleanArray>,
        points: MutableList<Pair<Int, Int>>
    ) {
        val queue = ArrayDeque<Pair<Int, Int>>()
        queue.add(Pair(startX, startY))
        visited[startX][startY] = true

        while (queue.isNotEmpty()) {
            val (x, y) = queue.removeFirst()
            points.add(Pair(x, y))

            for (dx in -1..1) {
                for (dy in -1..1) {
                    if (dx == 0 && dy == 0) continue
                    val nx = x + dx
                    val ny = y + dy

                    if (nx < 0 || nx >= bitmap.width || ny < 0 || ny >= bitmap.height) continue
                    if (visited[nx][ny]) continue
                    if (!isPixelBlack(bitmap, nx, ny)) continue

                    visited[nx][ny] = true
                    queue.add(Pair(nx, ny))
                }
            }
        }
    }

    private fun isPixelBlack(bitmap: Bitmap, x: Int, y: Int): Boolean {
        val pixel = bitmap.getPixel(x, y)
        val brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
        return brightness < 128
    }

    private fun getBounds(points: List<Pair<Int, Int>>): Rect {
        var minX = points.minOf { it.first }
        var maxX = points.maxOf { it.first }
        var minY = points.minOf { it.second }
        var maxY = points.maxOf { it.second }

        val padding = 8
        return Rect(
            maxOf(0, minX - padding),
            maxOf(0, minY - padding),
            minOf(bitmapWidth - 1, maxX + padding),
            minOf(bitmapHeight - 1, maxY + padding)
        )
    }

    private fun cropToBounds(bitmap: Bitmap, rect: Rect): Bitmap {
        return Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height())
    }
}

// ====== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======
private fun saveSymbol(strokes: List<List<Offset>>, symbol: String, context: Context) {
    val bitmap = createBitmapFromStrokes(strokes, context, Size(1000f, 1000f)) ?: return
    val centered = centerAndResize(bitmap, 64)
    val dir = File(context.getExternalFilesDir(null), "dataset/$symbol")
    dir.mkdirs()
    val file = File(dir, "${System.currentTimeMillis()}.png")

    FileOutputStream(file).use { out -> centered.compress(Bitmap.CompressFormat.PNG, 100, out) }
    centered.recycle()
    bitmap.recycle()
}

private fun centerAndResize(bitmap: Bitmap, size: Int): Bitmap {
    var minX = bitmap.width
    var minY = bitmap.height
    var maxX = 0
    var maxY = 0
    var hasPixel = false

    for (x in 0 until bitmap.width) {
        for (y in 0 until bitmap.height) {
            val pixel = bitmap.getPixel(x, y)
            val brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
            if (brightness < 128) {
                hasPixel = true
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }



    if (!hasPixel) return Bitmap.createScaledBitmap(bitmap, size, size, true)

    val padding = 10
    minX = maxOf(0, minX - padding)
    minY = maxOf(0, minY - padding)
    maxX = minOf(bitmap.width - 1, maxX + padding)
    maxY = minOf(bitmap.height - 1, maxY + padding)

    val cropped = Bitmap.createBitmap(bitmap, minX, minY, maxX - minX, maxY - minY)
    return Bitmap.createScaledBitmap(cropped, size, size, true)
}

private fun DrawScope.drawLineStroke(
    points: List<Offset>,
    color: androidx.compose.ui.graphics.Color = androidx.compose.ui.graphics.Color.Black,
    strokeWidth: Float = 20f
) {
    if (points.isEmpty()) return
    if (points.size == 1) {
        drawCircle(color = color, radius = strokeWidth * 0.8f, center = points[0])
        return
    }
    for (i in 0 until points.size - 1) {
        drawLine(color = color, start = points[i], end = points[i + 1], strokeWidth = strokeWidth, cap = StrokeCap.Round)
    }
}

private fun DrawScope.drawStrokePath(
    points: List<Offset>,
    color: androidx.compose.ui.graphics.Color = androidx.compose.ui.graphics.Color.Black,
    strokeWidth: Float = 20f
) {
    if (points.isEmpty()) return

    val path = Path().apply {
        moveTo(points.first().x, points.first().y)
        for (i in 1 until points.size) {
            lineTo(points[i].x, points[i].y)
        }
    }
    drawPath(
        path = path,
        color = color,
        style = Stroke(
            width = strokeWidth,
            cap = StrokeCap.Round
        )
    )
}
private fun createBitmapFromStrokes(strokes: List<List<Offset>>, context: Context, canvasSize: Size): Bitmap? {
    if (strokes.isEmpty()) return null

    val allPoints = strokes.flatten()
    if (allPoints.isEmpty()) return null

    val xs = allPoints.map { it.x }
    val ys = allPoints.map { it.y }
    var minX = xs.minOrNull() ?: return null
    var maxX = xs.maxOrNull() ?: return null
    var minY = ys.minOrNull() ?: return null
    var maxY = ys.maxOrNull() ?: return null

    val padding = 40f
    minX -= padding
    maxX += padding
    minY -= padding
    maxY += padding

    val width = maxX - minX
    val height = maxY - minY
    val side = maxOf(width, height).toInt()
    if (side <= 0) return null

    val bitmap = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)
    canvas.drawColor(Color.WHITE)

    val paint = Paint().apply {
        color = Color.BLACK
        strokeWidth = 15f
        style = Paint.Style.STROKE
        isAntiAlias = true
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
    }

    val offsetX = -minX + (side - width) / 2
    val offsetY = -minY + (side - height) / 2

    for (stroke in strokes) {
        if (stroke.size == 1) {
            val point = stroke[0]
            canvas.drawCircle(point.x + offsetX, point.y + offsetY, paint.strokeWidth / 2, paint)
            continue
        }
        if (stroke.size < 2) continue
        var prev = stroke[0]
        for (i in 1 until stroke.size) {
            val curr = stroke[i]
            canvas.drawLine(prev.x + offsetX, prev.y + offsetY, curr.x + offsetX, curr.y + offsetY, paint)
            prev = curr
        }
    }

    return bitmap
}

