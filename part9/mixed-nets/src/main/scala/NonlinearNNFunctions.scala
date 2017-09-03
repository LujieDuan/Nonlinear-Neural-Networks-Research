import Common._
import breeze.linalg.{DenseMatrix, DenseVector, sum}

/**
  * Created by LD on 2017-05-19.
  */
object NonlinearFunctions {
  /**
    *
    * @param z
    * @param t
    * @param w
    * @return
    */
  def transposeAndMultiply(z: DenseVector[Double], t: DenseVector[Double],
                           w: DenseMatrix[Seq[(Double, Double)]]): DenseMatrix[Seq[(Double, Double)]] = {
    w.mapPairs((index, s) => s.zipWithIndex.map {
      case (v, i) =>
        if(i == s.length - 1) (t(index._1) * v._2 * pow(z(index._2), v._1) * lan(z(index._2)), t(index._1) * pow(z(index._2), v._1))
        else (0.0, t(index._1) * pow(z(index._2), v._1))
    })
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def layerMultiply(z: DenseVector[Seq[(Double, Double)]], t: DenseVector[Double]): DenseVector[Double] = {
    assert(z.length == t.length)
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * pow(t(index), zvv._1))))
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def layerMultiplyPrime(z: DenseVector[Seq[(Double, Double)]], t: Double): DenseVector[Double] = {
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * zvv._1 * pow(t, zvv._1 - 1))))
  }

  /**
    *
    * @param z
    * @return
    */
  def cloneWeights(z: Seq[DenseMatrix[Seq[(Double, Double)]]]): Seq[DenseMatrix[Seq[(Double, Double)]]] = {
    z.map(m => m.map(s => s.map(_ => (0.0, 0.0))))
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def addDelta(z: DenseMatrix[Seq[(Double, Double)]], t: DenseMatrix[Seq[(Double, Double)]]): DenseMatrix[Seq[(Double, Double)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0, 0.0), (0.0, 0.0)).map(x => (x._1._1 + x._2._1, x._1._2 + x._2._2)))
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def minusDelta(z: DenseMatrix[Seq[(Double, Double)]], t: DenseMatrix[Seq[(Double, Double)]]): DenseMatrix[Seq[(Double, Double)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0, 0.0), (0.0, 0.0)).map(x => {
      if(x._1._1 - x._2._1 < 0) (0.0, x._1._2 - x._2._2) else (x._1._1 - x._2._1, x._1._2 - x._2._2)
    }))
  }

  /**
    *
    * @param z
    * @return
    */
  def addExponential(z: Seq[DenseMatrix[Seq[(Double, Double)]]]): Seq[DenseMatrix[Seq[(Double, Double)]]] = {
    z.map(w => w.map(s => {
      if(s.last._1 > s.length + 1) {
        val original_length = s.length
        val original_exp = s.last._1
        val original_coeff = s.last._2
        val updated = s.updated(original_length - 1, (original_length.toDouble, original_coeff))
        updated :+ (original_exp, 0.0)
      } else s
    }))
  }

  /**
    *
    * @param z
    */
  private def printWeights(z: Seq[DenseMatrix[Seq[(Double, Double)]]])(implicit fileName: String) = {
    z.zipWithIndex.foreach(ma => ma._1.foreachValue(se => log(s"Layer: ${ma._2 + 1} ${se.map(t => t._2.toString + "x^"+ t._1.toString + " ").mkString}")))
  }
}
