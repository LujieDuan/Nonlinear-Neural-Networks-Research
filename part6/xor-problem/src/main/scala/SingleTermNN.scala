import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.stats.distributions.Rand

/**
  * Created by LD on 2017-06-13.
  */
class SingleTermNN(learning_rate: Double, sizes: Seq[Int], epoch: Int, mini_batch_size: Int){

  var biases: Seq[DenseVector[Double]] = _

  var weights : Seq[DenseMatrix[Double]] = _

  var exponents: Seq[DenseMatrix[Double]] = _

  var layers: Int = 0

  def start(): Unit = {
    biases = sizes.drop(1).map(x => DenseVector.rand(x, rand = Rand.gaussian))
    weights = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2), rand = Rand.gaussian))
    exponents = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.fill[Double](x._1, sizes(x._2))(1.0))
    layers = sizes.length - 1
    val datasets = MnistLoader.load()
    val start = System.currentTimeMillis()
    val costs = initialCosts()

    println(s"Single Term Nonlinear Neural Network:${sizes.mkString("-")}")
    println(s"Batch Size: $mini_batch_size. Learning Rate: $learning_rate")

    SGD(datasets, costs, epoch, mini_batch_size, learning_rate, start, updateMiniBatch, feedForward)

    println(s"Total Time: ${(System.currentTimeMillis() - start) / 1000.0}s")
  }

  /**
    *
    * @param input
    * @return
    */
  def feedForward(input: DenseVector[Double]): DenseVector[Double] = {
    var output: DenseVector[Double] = input
    (biases, weights, exponents).zipped.foreach((b, w, e) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(w(x, ::).t *:* powVectorVector(output, e(x, ::).t))).toArray) + b))
    assert(output.length == sizes(layers))
    output
  }

  /**
    *
    * @param mini_batch
    * @param eta
    */
  def updateMiniBatch(mini_batch: Seq[(DenseVector[Double], DenseVector[Double])], eta: Double) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = weights.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))
    var nable_e = exponents.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))
    mini_batch.foreach(x => {
      val delta = backprop(x._1, x._2)
      nable_b = (nable_b, delta._1).zipped.map((nb, dnb) => nb+dnb)
      nable_w = (nable_w, delta._2).zipped.map((nw, dnw) => nw+dnw)
      nable_e = (nable_e, delta._3).zipped.map((ne, dne) => ne+dne)
    })
    biases = (biases, nable_b).zipped.map((b, nb) => b - nb.map(x => x * (eta/mini_batch.length)))
    weights = (weights, nable_w).zipped.map((w, nw) => w - nw.map(x => x * (eta/mini_batch.length)))
    exponents = (exponents, nable_e).zipped.map((e, ne) => e - ne.map(x => x * (eta/mini_batch.length)))
  }

  /**
    *
    * @param x
    * @param y
    * @return
    */
  def backprop(x: DenseVector[Double], y: DenseVector[Double]):
  (Seq[DenseVector[Double]], Seq[DenseMatrix[Double]], Seq[DenseMatrix[Double]]) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
    var nable_w = weights.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))
    var nable_e = exponents.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))
    var activation: DenseVector[Double] = x
    var activations: Seq[DenseVector[Double]] = Seq(activation)
    var zs: Seq[DenseVector[Double]] = Seq.empty
    (biases, weights, exponents).zipped.foreach((b, w, e) => {
      val z = DenseVector((0 until w.rows).map(x => sum(w(x, ::).t *:* powVectorVector(activation, e(x, ::).t))).toArray) + b
      zs = zs :+ z
      activation = sigmoid(z)
      activations = activations :+ activation
    })

    //backward pass
    var delta = costDerivative(activations(layers), y) * sigmoid_prime(zs(layers - 1))
    nable_b = nable_b.updated(layers - 1, delta)
    nable_w = nable_w.updated(layers - 1, errorWrtWeight(activations(layers - 1), exponents(layers - 1), delta))
    nable_e = nable_e.updated(layers - 1, errorWrtExponent(activations(layers - 1), weights(layers - 1), exponents(layers - 1), delta))

    //lower layers
    for(l <- 2 to layers) {
      val z = zs(layers - l)
      val sp = sigmoid_prime(z)
      val w = weights(layers - l + 1)
      val e = exponents(layers - l + 1)
      delta = DenseVector((0 until w.cols).map(x => calculateDelta(activations, delta, l, w, e, x)).toArray) * sp
      nable_b = nable_b.updated(layers - l, delta)
      nable_w = nable_w.updated(layers - l, errorWrtWeight(activations(layers - l), exponents(layers - l), delta))
      nable_e = nable_e.updated(layers - l, errorWrtExponent(activations(layers - l), weights(layers - l), exponents(layers - l), delta))
    }
    (nable_b, nable_w, nable_e)
  }

  /**
    *
    * @param activations
    * @param delta
    * @param l
    * @param w
    * @param e
    * @param x
    * @return
    */
  private def calculateDelta(activations: Seq[DenseVector[Double]], delta: DenseVector[Double], l: Int, w: DenseMatrix[Double], e: DenseMatrix[Double], x: Int) = {
    sum(w(::, x) *:* delta *:* e(::, x) *:* powNumVector(activations(layers - l + 1)(x), e(::, x) - 1.0))
  }

  /**
    *
    * @param activationOutput
    * @param originalExponents
    * @param delta
    * @return
    */
  def errorWrtWeight(activationOutput: DenseVector[Double], originalExponents: DenseMatrix[Double], delta: DenseVector[Double]): DenseMatrix[Double] = {
    assert(activationOutput.length == originalExponents.cols)
    assert(delta.length == originalExponents.rows)
    DenseMatrix.tabulate(originalExponents.rows, originalExponents.cols){
      case (i, j) => delta(i) * pow(activationOutput(j), originalExponents(i, j))
    }
  }

  /**
    *
    * @param activationOutput
    * @param originalWeights
    * @param originalExponents
    * @param delta
    * @return
    */
  def errorWrtExponent(activationOutput: DenseVector[Double], originalWeights: DenseMatrix[Double],
                       originalExponents: DenseMatrix[Double], delta: DenseVector[Double]): DenseMatrix[Double] = {
    assert(activationOutput.length == originalWeights.cols)
    assert(originalExponents.cols == originalWeights.cols)
    assert(originalExponents.rows == originalWeights.rows)
    assert(delta.length == originalWeights.rows)
    DenseMatrix.tabulate(originalWeights.rows, originalWeights.cols){
      case (i, j) => delta(i) * originalWeights(i, j) * pow(activationOutput(j), originalExponents(i, j)) * lan(activationOutput(j))
    }
  }
}
