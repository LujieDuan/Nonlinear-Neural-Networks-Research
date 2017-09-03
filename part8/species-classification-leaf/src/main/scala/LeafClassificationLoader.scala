import breeze.linalg.DenseVector

import scala.collection.mutable.ListBuffer

/**
  * Created by LD on 2017-08-13.
  */
object LeafClassificationLoader {

  private def LoadLines(fileName: String): Seq[Seq[(DenseVector[Double], DenseVector[Double])]] = {
    var contents = new ListBuffer[Seq[String]]
    val bufferedSource = io.Source.fromFile(fileName)
    for (line <- bufferedSource.getLines) {
      val cols = line.split(",")
      contents += cols
    }
    val maxValue = contents.flatMap(x => x.drop(1).map(_.toDouble)).max
    val scale = 1 / maxValue
    val groupedMap = contents.groupBy(_.head)
    val outputSize = groupedMap.size
    val resultMap = (0 until outputSize).zip(groupedMap).map(x => x._1 -> x._2._2)
    bufferedSource.close
    val result = resultMap.map(x => x._2.map(y => {
      val label = DenseVector.zeros[Double](outputSize)
      label(x._1) = 1
      (DenseVector(y.drop(1).map(x => x.toDouble * scale).toArray), label)
    }))
    result
  }

  def Load(dir: String, trainRatio: Double, validationRatio: Double): Common.SetsType = {
    val loaded = LoadLines(dir + "data_Mar_64.txt")
    val size = loaded.head.size
    val trainSize = (size * trainRatio).toInt
    val validationSize = (size * validationRatio).toInt
    new Common.SetsType(loaded.flatMap(_.take(trainSize)), loaded.flatMap(_.takeRight(size - trainSize).take(validationSize)), loaded.flatMap(_.takeRight(size - trainSize - validationSize)))
  }
}
