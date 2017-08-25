
import Common._
import LinearNNFunctions.transposeAndMultiply
import Pack.packWeights
import breeze.linalg.{DenseMatrix, DenseVector, sum}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-05-17.
  * Modified based on code from Python: https://github.com/mnielsen/neural-networks-and-deep-learning
  */
class LinearNN(learning_rate: Double, sizes: Seq[Int], epoch: Int, mini_batch_size: Int){

  val filePrefix = "Linear-MNIST"

  implicit var fileName = ""

  var biases: Seq[DenseVector[Double]] = _

  var weights : Seq[DenseMatrix[Double]] = _

  var layers: Int = 0

  /**
    *
    */
  def start(): Unit = {
    val hp = ParameterLoader.load(sizes)
    biases = hp._1
    weights = hp._2
    /*biases = sizes.drop(1).map(x => DenseVector.rand(x, rand = Rand.gaussian))
    weights = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2), rand = Rand.gaussian))*/
    layers = sizes.length - 1
    val datasets = MnistLoader.load()
    val start = System.currentTimeMillis()
    val costs = initialCosts()

    fileName = generateFileName(learning_rate, mini_batch_size, start, sizes, filePrefix)
    log(s"$filePrefix:${sizes.mkString("-")}")
    log(s"Batch Size: $mini_batch_size. Learning Rate: $learning_rate")

    SGD(datasets, costs, epoch, mini_batch_size, learning_rate, start, update_mini_batch, feedForward)

    outputResults(datasets, costs, start)
    packWeights(biases, weights, fileName)
  }

  /**
    *
    * @param input
    * @return
    */
  private def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
    var output :DenseVector[Double] = input
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(w(x, ::).t *:* output)).toArray) + b))
    assert(output.length == sizes(layers))
    output
  }


  /**
    *
    * @param mini_batch
    * @param eta
    */
  private def update_mini_batch(mini_batch: Seq[(DenseVector[Double], DenseVector[Double])], eta: Double) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = weights.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))
    mini_batch.foreach(x => {
      val delta = backprop(x._1, x._2)
//      val checkgrad = gradientCheck(x._1, x._2)
//      val deltaFlat = unroll(delta._1, delta._2)
//      val numgradFlat = unroll(checkgrad._1, checkgrad._2)
//      val difft = breeze.linalg.norm(deltaFlat - numgradFlat)
//      val diffb = breeze.linalg.norm(deltaFlat + numgradFlat)
//      val diff = difft / diffb
//      println(diff)
      nable_b = (nable_b, delta._1).zipped.map((nb, dnb) => nb+dnb)
      nable_w = (nable_w, delta._2).zipped.map((nw, dnw) => nw+dnw)
    })
    biases = (biases, nable_b).zipped.map((b, nb) => b - nb.map(x => x * (eta/mini_batch.length)))
    weights = (weights, nable_w).zipped.map((w, nw) => w - nw.map(x => x * (eta/mini_batch.length)))
  }

  /**
    *
    * @param x
    * @param y
    * @return
    */
  private def backprop(x: DenseVector[Double], y: DenseVector[Double]):
  (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = weights.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))
    var activation: DenseVector[Double] = x
    var activations: Seq[DenseVector[Double]] = Seq(activation)
    var zs: Seq[DenseVector[Double]] = Seq.empty
    (biases, weights).zipped.foreach((b, w) => {
      val z = DenseVector((0 until w.rows).map(x => sum(w(x, ::).t *:* activation)).toArray) + b
      zs = zs :+ z
      //val temp = (w * activation.toDenseMatrix.t) + b.toDenseMatrix.t
      activation = sigmoid(z)
      activations = activations :+ activation
    })
    //backward pass
    var delta = cost_derivative(activations(layers), y) * sigmoid_prime(zs(layers - 1))
    nable_b = nable_b.updated(layers - 1, delta)
    nable_w = nable_w.updated(layers - 1, transposeAndMultiply(activations(layers - 1), delta))

    //lower layers
    for(l <- 2 to layers) {
      val z = zs(layers - l)
      val sp = sigmoid_prime(z)
      val w = weights(layers - l + 1)
      delta = DenseVector((0 until w.cols).map(x => sum(w(::, x) *:* delta)).toArray) * sp
      nable_b = nable_b.updated(layers - l, delta)
      nable_w = nable_w.updated(layers - l, transposeAndMultiply(activations(layers - l), delta))
    }
    (nable_b, nable_w)
  }

  private def gradientCheck(x: DenseVector[Double], y: DenseVector[Double]):
  (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = weights.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))

    var ebl = 0.0001f

    for (i <- biases.indices) {
      biases(i).foreachPair((ind, v) => {
        biases(i)(ind) = v - ebl
        val out1 = feedForwardChecking(x, y)
        biases(i)(ind) = v + ebl
        val out2 = feedForwardChecking(x, y)
        nable_b(i)(ind) = (out2 - out1) / (2 * ebl)
        biases(i)(ind) = v
      })
    }

    for (i <- weights.indices) {
      weights(i).foreachPair((ind, v) => {
        weights(i)(ind) = v - ebl
        val out1 = feedForwardChecking(x, y)
        weights(i)(ind) = v + ebl
        val out2 = feedForwardChecking(x, y)
        nable_w(i)(ind) = (out2 - out1) / (2 * ebl)
        weights(i)(ind) = v
      })
    }

    (nable_b, nable_w)
  }

  /**
    *
    * @param input
    * @return
    */
  private def feedForwardChecking(input: DenseVector[Double], y: DenseVector[Double]): Double = {
    var output :DenseVector[Double] = input
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(w(x, ::).t *:* output)).toArray) + b))
    assert(output.length == sizes(layers))
    sum((output - y).map(x => pow(math.abs(x), 2.0f))) / 2.0f
  }

  private def unroll(bias: Seq[DenseVector[Double]], weight: Seq[DenseMatrix[Double]]): DenseVector[Double] = {
    var result: ArrayBuffer[Double] = ArrayBuffer()
    (bias, weight).zipped.foreach((b, w) => {
      result ++= b.toArray
      result ++= w.toArray
    })
    DenseVector(result.toArray)
  }

}

object LinearNNFunctions {

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def transposeAndMultiply(z: DenseVector[Double], t: DenseVector[Double]): DenseMatrix[Double] = {
    DenseMatrix.tabulate(t.length, z.length){case (i, j) => t(i) * z(j)}
  }
}
