import Common._
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.stats.distributions.Rand
import NonlinearFunctions._

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-05-19.
  */
class NonlinearNN(learning_rate: Double, sizes: Seq[Int], epoch: Int, mini_batch_size: Int) {

  var biases: Seq[DenseVector[Double]] = _

  var weights : Seq[DenseMatrix[Seq[(Double, Double)]]] = _

  var layers: Int = 0

  def start(): Unit = {

    // Include the following three lines and comment out the next two to read initial parameters from saved file
//    val hp = ParameterLoader.load(sizes)
//    biases = hp._1
//    weights = hp._2
    biases = sizes.drop(1).map(x => DenseVector.rand(x, rand = Rand.gaussian))
    weights = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2), rand = Rand.gaussian)).map(x => x.map(y => Seq[(Double, Double)]((1.0, y))))

      // Include the following two lines if want to save the parameters to repeat experiment
//    val temp = unrollWithoutExp(biases, weights)
//    println(temp)

    layers = sizes.length - 1
    val datasets = MnistLoader.load()
    val start = System.currentTimeMillis()
    val costs = initialCosts()

    println(s"Nonlinear Neural Network Structure:${sizes.mkString("-")}")
    println(s"Batch Size: $mini_batch_size. Learning Rate: $learning_rate")

    SGD(datasets, costs, epoch, mini_batch_size, learning_rate, start, updateMiniBatch, feedForward)

    println(s"Total Time: ${(System.currentTimeMillis() - start) / 1000.0}s")
  }

  /**
    *
    * @param input
    * @return
    */
  private def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
    var output :DenseVector[Double] = input
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, output))).toArray) + b))
    assert(output.length == sizes(layers))
    output
  }

  /**
    *
    * @param mini_batch
    * @param eta
    */
  private def updateMiniBatch(mini_batch: Seq[(DenseVector[Double], DenseVector[Double])], eta: Double) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = cloneWeights(weights)
    mini_batch.foreach(x => {
      val delta = backprop(x._1, x._2)
      nable_b = (nable_b, delta._1).zipped.map((nb, dnb) => nb+dnb)
      nable_w = (nable_w, delta._2).zipped.map((nw, dnw) => addDelta(nw, dnw))
    })
    biases = (biases, nable_b).zipped.map((b, nb) => b - nb.map(x => x * (eta/mini_batch.length)))
    weights = (weights, nable_w).zipped.map((w, nw) => minusDelta(w, nw.map(x => x.map(y => (y._1 * (eta/mini_batch.length), y._2 * (eta/mini_batch.length))))))
    weights = addExponential(weights)
  }

  /**
    *
    * @param x
    * @param y
    * @return
    */
  private def backprop(x: DenseVector[Double], y: DenseVector[Double]):
  (Seq[DenseVector[Double]], Seq[DenseMatrix[Seq[(Double, Double)]]]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = cloneWeights(weights)
    var activation: DenseVector[Double] = x
    var activations: Seq[DenseVector[Double]] = Seq(activation)
    var zs: Seq[DenseVector[Double]] = Seq.empty
    (biases, weights).zipped.foreach((b, w) => {
      val z = DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, activation))).toArray) + b
      zs = zs :+ z
      activation = sigmoid(z)
      activations = activations :+ activation
    })
    //backward pass
    var delta = cost_derivative(activations(layers), y) * sigmoid_prime(zs(layers - 1))
    nable_b = nable_b.updated(layers - 1, delta)
    nable_w = nable_w.updated(layers - 1, transposeAndMultiply(activations(layers - 1), delta, weights(layers - 1)))

    //lower layers
    for(l <- 2 to layers) {
      val z = zs(layers - l)
      val sp = sigmoid_prime(z)
      val w = weights(layers - l + 1).t
      delta = DenseVector((0 until w.rows).map(x => sum(layerMultiplyPrime(w(x, ::).t, activations(layers - l + 1)(x)) * delta)).toArray) * sp
      nable_b = nable_b.updated(layers - l, delta)
      nable_w = nable_w.updated(layers - l, transposeAndMultiply(activations(layers - l), delta, weights(layers - l)))
    }
    (nable_b, nable_w)
  }

  /**
    * Used for saving initial parameters
    * @param bias
    * @param weight
    * @return
    */
  private def unrollWithoutExp(bias: Seq[DenseVector[Double]], weight: Seq[DenseMatrix[Seq[(Double, Double)]]]): DenseVector[Double] = {
    var result: ArrayBuffer[Double] = ArrayBuffer()
    (bias, weight).zipped.foreach((b, w) => {
      result ++= b.toArray
      result ++= w.toArray.foldLeft(ArrayBuffer[Double]())((b, a) => b ++= a.toArray.foldLeft(ArrayBuffer[Double]())((c, d) => {
        c += d._2
      }))
    })
    DenseVector(result.toArray)
  }

}

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
}
