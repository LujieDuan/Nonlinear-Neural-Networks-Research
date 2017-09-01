import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by LD on 2017-08-04.
  */
object ParameterLoader {

  def loadLinear(layers: Seq[Int]): (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]]) = {
    val bufferedSource = io.Source.fromFile("./input/linear/hparam2-2-1.txt")
    val str = bufferedSource.getLines().next()
    val arr = str.replace('\'', ' ').split(", ").map(_.trim().toDouble)
    var cur = 0
    var bs: Seq[DenseVector[Double]] = Seq.empty
    var ws: Seq[DenseMatrix[Double]] = Seq.empty
    for (i <- 1 until layers.length){
      val b = new DenseVector(arr.slice(cur, cur + layers(i)))
      cur += layers(i)
      val w = new DenseVector(arr.slice(cur, cur + layers(i-1) * layers(i)))
      cur += layers(i-1) * layers(i)
      bs :+= b
      ws :+= w.toDenseMatrix.reshape(layers(i-1), layers(i)).t
    }
    (bs, ws)
  }

  def loadNonlinear(layers: Seq[Int]): (Seq[DenseVector[Double]], Seq[DenseMatrix[Seq[(Double, Double)]]]) = {
    val bufferedSource = io.Source.fromFile("./input/nonlinear/hparam2-1-1.txt")
    val str = bufferedSource.getLines().next()
    val arr = str.replace('\'', ' ').split(", ").map(_.trim().toDouble)
    var cur = 0
    var bs: Seq[DenseVector[Double]] = Seq.empty
    var ws: Seq[DenseMatrix[Seq[(Double, Double)]]] = Seq.empty
    for (i <- 1 until layers.length){
      val b = new DenseVector(arr.slice(cur, cur + layers(i)))
      cur += layers(i)
      val w = new DenseVector(arr.slice(cur, cur + layers(i-1) * layers(i))).map(x => Seq((1.0, x)))
      cur += layers(i-1) * layers(i)
      bs :+= b
      ws :+= w.toDenseMatrix.reshape(layers(i-1), layers(i)).t
    }
    (bs, ws)
  }

  def loadSingleTerm(layers: Seq[Int]): (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]]) = {
    val bufferedSource = io.Source.fromFile("./input/single-term/hparam2-2-1.txt")
    val str = bufferedSource.getLines().next()
    val arr = str.replace('\'', ' ').split(", ").map(_.trim().toDouble)
    var cur = 0
    var bs: Seq[DenseVector[Double]] = Seq.empty
    var ws: Seq[DenseMatrix[Double]] = Seq.empty
    for (i <- 1 until layers.length){
      val b = new DenseVector(arr.slice(cur, cur + layers(i)))
      cur += layers(i)
      val w = new DenseVector(arr.slice(cur, cur + layers(i-1) * layers(i)))
      cur += layers(i-1) * layers(i)
      bs :+= b
      ws :+= w.toDenseMatrix.reshape(layers(i-1), layers(i)).t
    }
    (bs, ws)
  }
}
