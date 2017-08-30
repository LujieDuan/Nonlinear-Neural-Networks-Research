import Common._
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.stats.distributions.Rand

/**
  * Created by LD on 2017-05-17.
  * Modified based on code from Python: https://github.com/mnielsen/neural-networks-and-deep-learning
  */
class LinearNN(learning_rate: Double, sizes: Seq[Int], epoch: Int, mini_batch_size: Int){

  var biases: Seq[DenseVector[Double]] = _

  var weights : Seq[DenseMatrix[Double]] = _

  var layers: Int = 0

  /**
    * Start the training
    */
  def start(): Unit = {
    biases = sizes.drop(1).map(x => DenseVector.rand(x, rand = Rand.gaussian))
    weights = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2), rand = Rand.gaussian))
    //Uncomment next three lines to read parameters from file instead of random initialize
    val hp = ParameterLoader.load(sizes)
    biases = hp._1
    weights = hp._2

    layers = sizes.length - 1
    val datasets = MnistLoader.load()
    val start = System.currentTimeMillis()
    val costs = initialCosts()
    println(s"Linear Neural Network Structure:${sizes.mkString("-")}")
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
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(w(x, ::).t *:* output)).toArray) + b))
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
    var nable_w = weights.map(x => DenseMatrix.zeros[Double](x.rows, x.cols))
    mini_batch.foreach(x => {
      val delta = backprop(x._1, x._2)
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
      val z = ((w * activation.toDenseMatrix.t) + b.toDenseMatrix.t).toDenseVector
      zs = zs :+ z
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

