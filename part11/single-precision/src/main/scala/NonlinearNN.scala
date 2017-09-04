import Common._
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.stats.distributions.Rand
import NonlinearFunctions._

/**
  * Created by LD on 2017-05-19.
  */
class NonlinearNN(learning_rate: Float, sizes: Seq[Int], epoch: Int, mini_batch_size: Int) {

  var biases: Seq[DenseVector[Float]] = _

  var weights : Seq[DenseMatrix[Seq[(Float, Float)]]] = _

  var layers: Int = 0

  def start(): Unit = {
    biases = sizes.drop(1).map(x => DenseVector.rand(x, rand = Rand.gaussian).map(_.toFloat))
    weights = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2), rand = Rand.gaussian)).map(x => x.map(y => Seq[(Float, Float)]((1.0f, y.toFloat))))
    layers = sizes.length - 1
    val datasets = MnistLoader.load()
    val start = System.currentTimeMillis()
    val costs = initialCosts()

    println(s"Nonlinear Neural Network:${sizes.mkString("-")}")
    println(s"Batch Size: $mini_batch_size. Learning Rate: $learning_rate")

    SGD(datasets, costs, epoch, mini_batch_size, learning_rate, start, update_mini_batch, feedForward)

    println(s"Total Time: ${System.currentTimeMillis() - start}ms")

  }

  /**
    *
    * @param input
    * @return
    */
  private def feedForward(input: DenseVector[Float]): DenseVector[Float] = {
    var output :DenseVector[Float] = input
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, output))).toArray) + b))
    assert(output.length == sizes(layers))
    output
  }

  /**
    *
    * @param mini_batch
    * @param eta
    */
  private def update_mini_batch(mini_batch: Seq[(DenseVector[Float], DenseVector[Float])], eta: Float) = {
    var nable_b = biases.map(x => DenseVector.zeros[Float](x.length))
    var nable_w = cloneWeights(weights)
    mini_batch.foreach(x => {
      val delta = backprop(x._1, x._2)
      nable_b = (nable_b, delta._1).zipped.map((nb, dnb) => nb+dnb)
      nable_w = (nable_w, delta._2).zipped.map((nw, dnw) => addDelta(nw, dnw))
    })
    biases = (biases, nable_b).zipped.map((b, nb) => b - nb.map(x => x * (eta/mini_batch.length)))
    weights = (weights, nable_w).zipped.map((w, nw) => minusDelta(w, nw.map(x => x.map(y => (y._1 * (eta/mini_batch.length), y._2 * (eta/mini_batch.length))))))
    //println(s"$batch")
    weights = addExponential(weights)
  }

  /**
    *
    * @param x
    * @param y
    * @return
    */
  private def backprop(x: DenseVector[Float], y: DenseVector[Float]):
  (Seq[DenseVector[Float]], Seq[DenseMatrix[Seq[(Float, Float)]]]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Float](x.length))
    var nable_w = cloneWeights(weights)
    var activation: DenseVector[Float] = x
    var activations: Seq[DenseVector[Float]] = Seq(activation)
    var zs: Seq[DenseVector[Float]] = Seq.empty
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

    //
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


}

object NonlinearFunctions {
  /**
    *
    * @param z
    * @param t
    * @param w
    * @return
    */
  def transposeAndMultiply(z: DenseVector[Float], t: DenseVector[Float],
                           w: DenseMatrix[Seq[(Float, Float)]]): DenseMatrix[Seq[(Float, Float)]] = {
    w.mapPairs((index, s) => s.zipWithIndex.map {
      case (v, i) =>
        if(i == s.length - 1) (t(index._1) * v._2 * pow(z(index._2), v._1) * lan(z(index._2)), t(index._1) * pow(z(index._2), v._1))
        else (0.0f, t(index._1) * pow(z(index._2), v._1))
    })
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def layerMultiply(z: DenseVector[Seq[(Float, Float)]], t: DenseVector[Float]): DenseVector[Float] = {
    assert(z.length == t.length)
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * pow(t(index), zvv._1))))
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def layerMultiplyPrime(z: DenseVector[Seq[(Float, Float)]], t: Float): DenseVector[Float] = {
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * zvv._1 * pow(t, zvv._1 - 1))))
  }

  /**
    *
    * @param z
    * @return
    */
  def cloneWeights(z: Seq[DenseMatrix[Seq[(Float, Float)]]]): Seq[DenseMatrix[Seq[(Float, Float)]]] = {
    z.map(m => m.map(s => s.map(_ => (0.0f, 0.0f))))
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def addDelta(z: DenseMatrix[Seq[(Float, Float)]], t: DenseMatrix[Seq[(Float, Float)]]): DenseMatrix[Seq[(Float, Float)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0f, 0.0f), (0.0f, 0.0f)).map(x => (x._1._1 + x._2._1, x._1._2 + x._2._2)))
  }

  /**
    *
    * @param z
    * @param t
    * @return
    */
  def minusDelta(z: DenseMatrix[Seq[(Float, Float)]], t: DenseMatrix[Seq[(Float, Float)]]): DenseMatrix[Seq[(Float, Float)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0f, 0.0f), (0.0f, 0.0f)).map(x => {
      if(x._1._1 - x._2._1 < 0) (0.0f, x._1._2 - x._2._2) else (x._1._1 - x._2._1, x._1._2 - x._2._2)
    }))
  }

  /**
    *
    * @param z
    * @return
    */
  def addExponential(z: Seq[DenseMatrix[Seq[(Float, Float)]]]): Seq[DenseMatrix[Seq[(Float, Float)]]] = {
    z.map(w => w.map(s => {
      if(s.last._1 > s.length + 1) {
        val original_length = s.length
        val original_exp = s.last._1
        val original_coeff = s.last._2
        val updated = s.updated(original_length - 1, (original_length.toFloat, original_coeff))
        updated :+ (original_exp, 0.0f)
      } else s
    }))
  }
}
