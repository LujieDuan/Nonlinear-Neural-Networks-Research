import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.DenseVector

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-06-05.
  */
object LeafCountDataLoader {



  private def LoadLabelsFromCSV(fileName: String, idCol: Int, countCol: Int): Map[String, Int] = {
    var result = Map[String, Int]()
    val bufferedSource = io.Source.fromFile(fileName)
    for (line <- bufferedSource.getLines) {
      val cols = line.split(",").map(_.trim)
      result += (cols(idCol) -> cols(countCol).toInt)
    }
    bufferedSource.close
    result
  }

  private def LoadRGBImage(fileName: String): Array[Double] = {
    val originalPhoto = ImageIO.read(new File(fileName))
    val photo = Scale(originalPhoto, 128, 128)
    val result = ArrayBuffer[Double]()
    for(i <- 0 until 128; j <- 0 until 128){
      val rgb = new Color(photo.getRGB(i, j))
      result += rgb.getRed.toDouble / 255
      result += rgb.getGreen.toDouble / 255
      result += rgb.getBlue.toDouble / 255
    }
    result.toArray
  }

  def LoadImages(dir: String, ratio: Double): (DenseVector[(DenseVector[Double], Double)], DenseVector[(DenseVector[Double], Double)], Int) = {
    val labels = LoadLabelsFromCSV(dir + "Leaf_counts.csv", 0, 1)
    val LeafCountMax = labels.maxBy(_._2)._2
    val size = labels.size
    val trainingSize = (size * ratio).toInt
    val results = labels.map(x => (DenseVector(LoadRGBImage(s"$dir${x._1}_rgb.png")), x._2.toDouble / LeafCountMax)).toArray
    (DenseVector(results.take(trainingSize)), DenseVector(results.takeRight(size - trainingSize)), LeafCountMax)
  }

  private def Scale(source: BufferedImage, dWidth: Int, dHeight: Int): BufferedImage = {
    val result = new BufferedImage(dWidth, dHeight, source.getType)
    val g = result.createGraphics()
    g.drawImage(source, 0, 0, dWidth, dHeight, null)
    g.dispose()
    result
  }

}
