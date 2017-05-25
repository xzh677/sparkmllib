package me.xinzhang.spark.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

/**
  * Created by xin on 24/05/2017.
  */
object DimensionReduction {

  Logger.getLogger("org").setLevel(Level.ERROR)
  val sc = new SparkContext("local[*]", "DimensionReduction")


  import java.awt.image.BufferedImage
  import javax.imageio.ImageIO
  import java.io.File


  def extractPixels(path: String, width: Int, height: Int): Array[Double] = {

    def loadImageFromFile(path: String): BufferedImage = {
      ImageIO.read(new File(path))
    }

    //convert the image from RGB to Grayscale with $width and $height size
    def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
      val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
      val g = bwImage.getGraphics()
      g.drawImage(image, 0, 0, width, height, null)
      g.dispose()
      bwImage
    }

    //reshape the image from matrix [width, height] to vector [width * height]
    def getPixelsFromImage(image: BufferedImage) : Array[Double] = {
      val width = image.getWidth
      val height = image.getHeight
      val pixels = Array.ofDim[Double](width * height)
      image.getData.getPixels(0, 0, width, height, pixels)
    }

    def writeImage(image: BufferedImage, imageType: String, path: String) = {
      ImageIO.write(image, imageType, new File(path))
    }

    def imageProcessingTest(): Unit = {
      val aePath = "data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
      val aeImage = loadImageFromFile(aePath)
      val grayImage = processImage(aeImage, 100, 100)
      writeImage(grayImage, "jpg", "python/Aaron_Eckhart_0001.jpg")
    }

    val raw = loadImageFromFile(path)
    val processed = processImage(raw, width, height)
    getPixelsFromImage(processed)
  }


  def main(args: Array[String]): Unit = {
    val path = "data/lfw/*"
    val rdd = sc.wholeTextFiles(path)
    val files = rdd.map { case (fileName, content) =>
      fileName.replace("file:", "")
    }
    val pixels = files.map(f => extractPixels(f, 50, 50))
    //println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))

    import org.apache.spark.mllib.linalg.Vectors
    val vectors = pixels.map(p => Vectors.dense(p))
    vectors.setName("image-vectors")
    vectors.cache()

    //normalise data
    import org.apache.spark.mllib.feature.StandardScaler
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    val scaledVectors = vectors.map(v => scaler.transform(v))

    //compute principal components
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    val pc = matrix.computePrincipalComponents(K)

    import breeze.linalg.DenseMatrix
    val pcBreeze = new DenseMatrix(pc.numRows, pc.numCols, pc.toArray)
    import breeze.linalg.csvwrite
    csvwrite(new File("python/pc.csv"), pcBreeze)

    val projected = matrix.multiply(pc)
    println(projected.numRows(), projected.numCols())

    //svd V matrix is equal to principal components
    val svd = matrix.computeSVD(K, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols}")
    println(s"S dimension: (${svd.s.size})")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols}")

    def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
      val bools = array1.zip(array2).map {
        case (v1, v2) =>
          if (math.abs(math.abs(v1) - math.abs(v2)) < 1e-6) true else false
      }
      bools.fold(true)(_ & _)
    }
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0,2.0,3.0)))
    println(approxEqual(Array(1.0,2.0,3.0), Array(3.0,2.0,1.0)))
    println("PC matrix is equal to SVD V matrix: " + approxEqual(pc.toArray, svd.V.toArray))

    val svd300 = matrix.computeSVD(300, computeU = false)
    val sMatrix = new DenseMatrix(1, 300, svd300.s.toArray)
    csvwrite(new File("python/s.csv"), sMatrix)
  }

}
