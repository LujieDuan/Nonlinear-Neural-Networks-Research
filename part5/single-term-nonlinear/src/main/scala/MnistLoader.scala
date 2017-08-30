/**
  * Created by LD on 2017-05-17.
  * Those functions are modified based on:
  * https://github.com/alno/scalann/blob/master/examples/src/main/scala/org/scalann/examples/Mnist.scala
  */
import java.io.{DataInputStream, File, FileInputStream, FileOutputStream}
import java.net.URL
import java.nio.channels.Channels
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream

import breeze.linalg._

import scala.collection.mutable.ListBuffer

/**
  *
  * @param location
  * @param fileName
  * https://github.com/alno/scalann/blob/master/examples/src/main/scala/org/scalann/examples/Mnist.scala
  */
class MnistFileReader(location: String, fileName: String) {

  private[this] val path = Paths.get(location, fileName)

  if (!Files.exists(path))
    download

  protected[this] val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(path.toString)))

  private def download: Unit = {
    val rbc = Channels.newChannel(new URL(s"http://yann.lecun.com/exdb/mnist/$fileName").openStream())
    val fos = new FileOutputStream(s"$location/$fileName")
    fos.getChannel.transferFrom(rbc, 0, Long.MaxValue)
  }

}

/**
  * https://github.com/alno/scalann/blob/master/examples/src/main/scala/org/scalann/examples/Mnist.scala
  * @param location
  * @param fileName
  */
class MnistLabelReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()

  val labelsAsInts = readLabels
  val labelsAsVectors = labelsAsInts.map { label =>
    DenseVector.tabulate[Double](10) { i => if (i == label) 1.0 else 0.0 }
  }

  private[this] def readLabels: Seq[Int] = {
    val result: ListBuffer[Int] = new ListBuffer[Int]
    for (_ <- 0 until count) {
      result += stream.readByte()
    }
    result
  }
}

/**
  * https://github.com/alno/scalann/blob/master/examples/src/main/scala/org/scalann/examples/Mnist.scala
  * @param location
  * @param fileName
  */
class MnistImageReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count = stream.readInt()
  val width = stream.readInt()
  val height = stream.readInt()

  val imagesAsMatrices = readImages
  val imagesAsVectors = imagesAsMatrices map { image =>
    DenseVector.tabulate(width * height) { i => image(i / width, i % height) / 256.0 }
  }

  private[this] def readImages: Seq[DenseMatrix[Int]] = {
    val result: ListBuffer[DenseMatrix[Int]] = new ListBuffer[DenseMatrix[Int]]
    for (_ <- 0 until count) {
      result += readImage()
    }
    result
  }

  private[this] def readImage(): DenseMatrix[Int] = {
    val m = DenseMatrix.zeros[Int](height, width)
    for (y <- 0 until height; x <- 0 until width)
      m(y, x) = stream.readUnsignedByte()
    m
  }

}

/**
  * @param location
  * @param name
  * https://github.com/alno/scalann/blob/master/examples/src/main/scala/org/scalann/examples/Mnist.scala
  * http://yann.lecun.com/exdb/mnist/
  */
class MnistDataset(location: String, name: String) {

  lazy val imageReader = new MnistImageReader(location, s"$name-images-idx3-ubyte.gz")
  lazy val labelReader = new MnistLabelReader(location, s"$name-labels-idx1-ubyte.gz")

  def imageWidth = imageReader.width
  def imageHeight = imageReader.height

  def imagesAsMatrices = imageReader.imagesAsMatrices
  def imagesAsVectors = imageReader.imagesAsVectors

  def labelsAsInts = labelReader.labelsAsInts
  def labelsAsVectors = labelReader.labelsAsVectors

  def dataset = imagesAsVectors zip labelsAsVectors

}

object MnistLoader {

  /**
    * Load train/validation/testSet sets, with default rate of train:validation = 90:10
    * @return (train, validation, testSet)
    */
  def load(): DataSet = loadWithRate(0.83334)

  /**
    * Load train/validation/testSet sets, with default rate of train:validation = rate
    * @param rate split rate
    * @return (train, validation, testSet)
    */
  def loadWithRate(rate: Double): DataSet = {

    val directory = "data/mnistdata"
    val locationFile = new File(directory)

    if (!locationFile.exists)
      locationFile.mkdirs

    val trainDataset = new MnistDataset(directory, "train").dataset
    val splitIndex = (trainDataset.length * rate).toInt
    val trainAndValidation = trainDataset.splitAt(splitIndex)
    val testDataset = new MnistDataset(directory, "t10k").dataset

    new DataSet(trainAndValidation._1, trainAndValidation._2, testDataset)
  }
}
