import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.stats.distributions.Rand
import Common._
import NonlinearFunctions._
import LinearNNFunctions._

/**
  * Created by LD on 2017-05-29.
  */
class LinearNonlinearMixed(learning_rate: Double, sizes: Seq[Int], epoch: Int, mini_batch_size: Int) {

  val filePrefix = "LNMix-MNIST"

  implicit var fileName = ""

  var biases: Seq[DenseVector[Double]] = _

  var weights : Seq[DenseMatrix[Seq[(Double, Double)]]] = _

  var layers: Int = 0

  var linearBiases: DenseVector[Double] = _

  var linearWeights: DenseMatrix[Double] = _

  def start(): Unit = {
    biases = sizes.drop(2).map(x => DenseVector.rand(x, rand = Rand.gaussian))
    weights = sizes.drop(2).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2 + 1), rand = Rand.gaussian)).map(x => x.map(y => Seq[(Double, Double)]((1.0, y))))
    linearBiases = DenseVector.rand(sizes(1), rand = Rand.gaussian)
    linearWeights = DenseMatrix.rand(sizes(1), sizes(0), rand = Rand.gaussian)
    layers = sizes.length - 1
    val datasets = MnistLoader.load()
    val start = System.currentTimeMillis()
    val costs = initialCosts()

    fileName = generateFileName(learning_rate, mini_batch_size, start, sizes, filePrefix)
    log(s"$filePrefix:${sizes.mkString("-")}")
    log(s"Batch Size: $mini_batch_size. Learning Rate: $learning_rate")

    SGD(datasets, costs, epoch, mini_batch_size, learning_rate, start, update_mini_batch, feedForward)

    outputResults(datasets, costs, start)
  }

  /**
    *
    * @param input
    * @return
    */
  private def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
    var output :DenseVector[Double] = input
    output = sigmoid(DenseVector((0 until linearWeights.rows).map(x => sum(linearWeights(x, ::).t *:* output)).toArray) + linearBiases)
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, output))).toArray) + b))
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
    var nable_w = cloneWeights(weights)
    var nable_linearb = DenseVector.zeros[Double](linearBiases.length)
    var nable_linearw = DenseMatrix.zeros[Double](linearWeights.rows, linearWeights.cols)
    mini_batch.foreach(x => {
      val delta = backprop(x._1, x._2)
      nable_b = (nable_b, delta._1).zipped.map((nb, dnb) => nb+dnb)
      nable_w = (nable_w, delta._2).zipped.map((nw, dnw) => addDelta(nw, dnw))
      nable_linearb += delta._3
      nable_linearw += delta._4
    })
    biases = (biases, nable_b).zipped.map((b, nb) => b - nb.map(x => x * (eta/mini_batch.length)))
    weights = (weights, nable_w).zipped.map((w, nw) => minusDelta(w, nw.map(x => x.map(y => (y._1 * (eta/mini_batch.length), y._2 * (eta/mini_batch.length))))))
    linearBiases = linearBiases - nable_linearb.map(x => x * (eta/mini_batch.length))
    linearWeights = linearWeights - nable_linearw.map(x => x * (eta/mini_batch.length))
    weights = addExponential(weights)
  }

  /**
    *
    * @param x
    * @param y
    * @return
    */
  private def backprop(x: DenseVector[Double], y: DenseVector[Double]):
  (Seq[DenseVector[Double]], Seq[DenseMatrix[Seq[(Double, Double)]]], DenseVector[Double], DenseMatrix[Double]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = cloneWeights(weights)
    var nable_linearb = DenseVector.zeros[Double](linearBiases.length)
    var nable_linearw = DenseMatrix.zeros[Double](linearWeights.rows, linearWeights.cols)
    var activation: DenseVector[Double] = x
    var activations: Seq[DenseVector[Double]] = Seq(activation)
    var zs: Seq[DenseVector[Double]] = Seq.empty


    val linearz = DenseVector((0 until linearWeights.rows).map(x => sum(linearWeights(x, ::).t *:* activation)).toArray) + linearBiases
    zs = zs :+ linearz
    activation = sigmoid(linearz)
    activations = activations :+ activation

    (biases, weights).zipped.foreach((b, w) => {
      val z = DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, activation))).toArray) + b
      zs = zs :+ z
      activation = sigmoid(z)
      activations = activations :+ activation
    })
    //backward pass
    var delta = cost_derivative(activations(layers), y) * sigmoid_prime(zs(layers - 1))
    if (layers == 1) {
      nable_linearb = delta
      nable_linearw = LinearTransposeAndMultiply(activations(0), delta)
      return (nable_b, nable_w, nable_linearb, nable_linearw)
    }

    nable_b = nable_b.updated(layers - 2, delta)
    nable_w = nable_w.updated(layers - 2, transposeAndMultiply(activations(layers - 1), delta, weights(layers - 2)))

    for(l <- 2 to layers) {
      val z = zs(layers - l)
      val sp = sigmoid_prime(z)
      val w = weights(layers - l).t
      delta = DenseVector((0 until w.rows).map(x => sum(layerMultiplyPrime(w(x, ::).t, activations(layers - l + 1)(x)) * delta)).toArray) * sp
      if (l == layers) {
        nable_linearb = delta
        nable_linearw = LinearTransposeAndMultiply(activations(0), delta)
      } else {
        nable_b = nable_b.updated(layers - l - 1, delta)
        nable_w = nable_w.updated(layers - l - 1, transposeAndMultiply(activations(layers - l), delta, weights(layers - l - 1)))
      }
    }

    (nable_b, nable_w, nable_linearb, nable_linearw)
  }
}