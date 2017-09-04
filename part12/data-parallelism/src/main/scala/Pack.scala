import java.io._

import Common.CostsType
import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by LD on 2017-07-12.
  */
object Pack {

  def packCosts(obj: CostsType, fileName: String) = {
    val directory = "pickle/"
    val locationFile = new File(directory)

    if (!locationFile.exists)
      locationFile.mkdirs

    var packed = "train\n" + obj._1.toArray.map(x => s"${x._1}: ${x._2}\n").mkString
    packed += "validation\n"  + obj._2.toArray.map(x => s"${x._1}: ${x._2}\n").mkString
    packed += "test\n" + obj._3.toArray.map(x => s"${x._1}: ${x._2}\n").mkString
    val fos = new PrintWriter(new FileOutputStream(new File(s"$directory$fileName.costs" ), true))
    fos.write(packed)
    fos.close()
  }

  def packWeights(biases: Seq[DenseVector[Float]], weights: Seq[DenseMatrix[Float]], fileName: String) = {
    val directory = "pickle/"
    val locationFile = new File(directory)

    if (!locationFile.exists)
      locationFile.mkdirs

    var packed = "bias\n" + biases.map(x => vectorToString(x)).mkString
    packed += "weights\n" + weights.map(x => matrixToString(x)).mkString
    val fos = new PrintWriter(new FileOutputStream(new File(s"$directory$fileName.parameters" ), true))
    fos.write(packed)
    fos.close()
  }

  def packWeights(biases: Seq[DenseVector[Float]], weights: Seq[DenseMatrix[Float]], exponents: Seq[DenseMatrix[Float]], fileName: String) = {
    val directory = "pickle/"
    val locationFile = new File(directory)

    if (!locationFile.exists)
      locationFile.mkdirs

    var packed = "bias\n" + biases.map(x => vectorToString(x)).mkString
    packed += "weights\n" + weights.map(x => matrixToString(x)).mkString
    packed += "exp\n" + exponents.map(x => matrixToString(x)).mkString
    val fos = new PrintWriter(new FileOutputStream(new File(s"$directory$fileName.parameters" ), true))
    fos.write(packed)
    fos.close()
  }

  def packWeightsExp(biases: Seq[DenseVector[Float]], weights: Seq[DenseMatrix[Seq[(Float, Float)]]], fileName: String) = {
    val directory = "pickle/"
    val locationFile = new File(directory)

    if (!locationFile.exists)
      locationFile.mkdirs

    var packed = "bias\n" + biases.map(x => vectorToString(x)).mkString
    packed += "weights\n" + weights.map(x => matrixToStringExp(x)).mkString
    val fos = new PrintWriter(new FileOutputStream(new File(s"$directory$fileName.parameters" ), true))
    fos.write(packed)
    fos.close()
  }

  def matrixToString(m: DenseMatrix[Float]): String = {
    m.mapPairs((index, value) => s"${index._1} ${index._2} $value").toArray.mkString("\n")
  }

  def vectorToString(m: DenseVector[Float]): String = {
    m.mapPairs((index, value) => s"$index $value").toArray.mkString("\n")
  }

  def matrixToStringExp(m: DenseMatrix[Seq[(Float, Float)]]): String = {
    m.mapPairs((index, value) => {
      val inner = value.map(x => s"${x._1}: ${x._2}").mkString("\n")
      s"${index._1} ${index._2} $inner"
    }).toArray.mkString("\n")
  }
}
