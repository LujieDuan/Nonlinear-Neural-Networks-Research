import java.sql.Timestamp

import breeze.linalg.{DenseMatrix, DenseVector, max, min, shuffle, sum}
import breeze.numerics.sqrt
import breeze.plot.{Figure, plot}
import breeze.stats.distributions.Rand
import breeze.stats.meanAndVariance

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-05-19.
  */
class NonlinearNNRegression(learning_rate: Double, sizes: Seq[Int], epoch: Int, mini_batch_size: Int) {

  var biases: Seq[DenseVector[Double]] = _

  var weights : Seq[DenseMatrix[Seq[(Double, Double)]]] = _

  var layers: Int = 0

  var leafCountMax: Int = 0

  var cost_plot: ArrayBuffer[(Long, Double)] = new ArrayBuffer[(Long, Double)]()

  var cost_plot_test: ArrayBuffer[(Long, Double)] = new ArrayBuffer[(Long, Double)]()

  var startTime: Long = _

  def start(): Unit = {
    println(s"Nonlinear Neural Network:${sizes.toString()}")
    biases = sizes.drop(1).map(x => DenseVector.rand(x, rand = Rand.gaussian))
    weights = sizes.drop(1).zipWithIndex.map(x => DenseMatrix.rand(x._1, sizes(x._2), rand = Rand.gaussian)).map(x => x.map(y => Seq[(Double, Double)]((1.0, y))))
    layers = sizes.length - 1
    val leafSets = LeafCountDataLoader.LoadImages("data/leaf/data/Ara2013-Canon/", 0.8f)
    val training = leafSets._1
    val test = leafSets._2
    leafCountMax = leafSets._3
    val epoch = 200
    val mini_batch_size = 1
    startTime = System.currentTimeMillis()
    println(s"Batch Size: $mini_batch_size. Learning Rate: $learning_rate")
    SGD(training, epoch, mini_batch_size, learning_rate, test)

    val timestamp = new Timestamp(startTime)
    val f = Figure(s"Nonlinear-leaf-${sizes.mkString("-")}-batch-$mini_batch_size-rate-$learning_rate-$timestamp.png")
    val p = f.subplot(0)
    p += plot(cost_plot.map(x => x._1/1000.0), cost_plot.map(x => x._2))
    p += plot(cost_plot_test.map(x => x._1/1000.0), cost_plot_test.map(x => x._2), '.')
    p.xlabel = "time in s"
    p.ylabel = "cost"
    f.saveas(s"Nonlinear-leaf-${sizes.mkString("-")}-batch-$mini_batch_size-rate-$learning_rate-$timestamp.png")
    println(s"Final costs on training-set: ${cost_plot.last._2 / training.length} test-set: ${cost_plot_test.last._2 / test.length}")
    println(s"Total Time: ${System.currentTimeMillis() - startTime}ms")
  }


  def feedForward(input: DenseVector[Double]): Double = {
    var output :DenseVector[Double] = input
    (biases, weights).zipped.foreach((b, w) => output = sigmoid(DenseVector((0 until w.rows).map(x => sum(layerMultiply(w(x, ::).t, output))).toArray) + b))
    assert(output.length == sizes(layers))
    sum(output)
  }

  def SGD(training_data: DenseVector[(DenseVector[Double], Double)],
          epochs: Int, mini_batch_size: Int, eta: Double,
          test_data: DenseVector[(DenseVector[Double], Double)]) = {
    var mini_batches: Seq[DenseVector[(DenseVector[Double], Double)]] = Seq.empty
    for(j <- 0 until epochs) {
      mini_batches = shuffle(training_data).toArray.grouped(mini_batch_size).map(x => DenseVector(x)).toSeq
      mini_batches.foreach(x => update_mini_batch(x, eta))

      val training_results = evaluate(training_data)
      val test_results = evaluate(test_data)
      cost_plot += training_results._2
      cost_plot_test += test_results._2
      println(s"Epoch $j: Traning Set:     ${training_results._1} / ${training_data.length} Stat: ${evaluateStat(training_data)}")
      println(s"         Testing Set:     ${test_results._1} / ${test_data.length} Stat: ${evaluateStat(test_data)}")
      //printWeights(weights)
    }
  }

  var batch = 0

  def update_mini_batch(mini_batch: DenseVector[(DenseVector[Double], Double)], eta: Double) = {
    var nable_b = biases.map(x => DenseVector.zeros[Double](x.length))
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
    batch += 1
  }

  def backprop(x: DenseVector[Double], y: Double):
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
    var delta = cost_derivative(sum(activations(layers)), y) * sigmoid_prime(zs(layers - 1))
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

  def evaluate(test_data: DenseVector[(DenseVector[Double], Double)]): (Int, (Long, Double)) = {
    val test_result_vector = test_data.map(x => (feedForward(x._1), x._2))
    val cost = (System.currentTimeMillis() - startTime, computeLoss(test_result_vector))
    (test_result_vector.toArray.count(x => math.abs(x._1 - x._2) < 0.15), cost)
  }

  def evaluateStat(test_data: DenseVector[(DenseVector[Double], Double)]): String = {
    val test_result = test_data.map(x => (feedForward(x._1), x._2))
    val differences = test_result.map(x => (x._1 - x._2) * leafCountMax)
    val errors = test_result.map(x => math.abs(x._1 - x._2) * leafCountMax)
    val meanVariance = meanAndVariance(errors)
    "Mean: " + meanVariance.mean + " Standard Variance: " + sqrt(meanVariance.variance) + " Max Difference: " + max(differences) + " Min Difference: " + min(differences)
  }


  def cost_derivative(output_activation: Double, y: Double): Double = {
    output_activation - y
  }

  def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    z.map(x => breeze.numerics.sigmoid(x))
  }

  def sigmoid_prime(z: DenseVector[Double]): DenseVector[Double] = {
    z.map(el => breeze.numerics.sigmoid(el) * (1 - breeze.numerics.sigmoid(el)))
  }

  def transposeAndMultiply(z: DenseVector[Double], t: DenseVector[Double],
                           w: DenseMatrix[Seq[(Double, Double)]]): DenseMatrix[Seq[(Double, Double)]] = {
    w.mapPairs((index, s) => s.zipWithIndex.map {
      case (v, i) =>
        if(i == s.length - 1) (t(index._1) * v._2 * pow(z(index._2), v._1) * lan(z(index._2)), t(index._1) * pow(z(index._2), v._1))
        else (0.0, t(index._1) * pow(z(index._2), v._1))
    })
  }


  def layerMultiply(z: DenseVector[Seq[(Double, Double)]], t: DenseVector[Double]): DenseVector[Double] = {
    assert(z.length == t.length)
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * pow(t(index), zvv._1))))
  }

  def layerMultiplyPrime(z: DenseVector[Seq[(Double, Double)]], t: Double): DenseVector[Double] = {
    z.mapPairs((index, zv) => sum(zv.map(zvv => zvv._2 * zvv._1 * pow(t, zvv._1 - 1))))
  }

  def cloneWeights(z: Seq[DenseMatrix[Seq[(Double, Double)]]]): Seq[DenseMatrix[Seq[(Double, Double)]]] = {
    z.map(m => m.map(s => s.map(_ => (0.0, 0.0))))
  }

  def addDelta(z: DenseMatrix[Seq[(Double, Double)]], t: DenseMatrix[Seq[(Double, Double)]]): DenseMatrix[Seq[(Double, Double)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0, 0.0), (0.0, 0.0)).map(x => (x._1._1 + x._2._1, x._1._2 + x._2._2)))
  }

  def minusDelta(z: DenseMatrix[Seq[(Double, Double)]], t: DenseMatrix[Seq[(Double, Double)]]): DenseMatrix[Seq[(Double, Double)]] = {
    assert(z.cols == t.cols && z.rows == t.rows)
    z.mapPairs((index, zv) => zv.zipAll(t(index._1, index._2), (0.0, 0.0), (0.0, 0.0)).map(x => {
      if(x._1._1 - x._2._1 < 0) (0.0, x._1._2 - x._2._2)else (x._1._1 - x._2._1, x._1._2 - x._2._2)
    }))
  }

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

  val Precision = 0.000000000001f

  /**
    * Compare Doubles with precision
    */
  def ~=(x: Double, y: Double, precision: Double = Precision): Boolean = {
    if ((x - y).abs < precision) true else false
  }

  /**
    * Returns the value ln(i) when i is not 0. Otherwise return -Integer.MAX_VAL
    */
  def lan(i: Double) : Double = if (~=(i, 0)) -Integer.MAX_VALUE else scala.math.log(i)

  /**
    * Returns the por power of num
    */
  def pow(num: Double, por: Double): Double = {
    assert(num >= 0)
    if (~=(por, 0) && ! ~=(num, 0))
      1
    else if (~=(por, 1.0))
      num
    else if (~=(num, 0))
      0
    else if (por < 0)
      1/scala.math.exp(lan(num)*(-por))
    else
      scala.math.exp(lan(num)*por)
  }

  def printWeights(z: Seq[DenseMatrix[Seq[(Double, Double)]]]) = {
    z.zipWithIndex.foreach(ma => ma._1.foreachValue(se => println(s"Layer: ${ma._2 + 1} ${se.map(t => t._2.toString + "x^"+ t._1.toString + " ").mkString}")))
  }

  def computeLoss(results: DenseVector[(Double, Double)]): Double = {
    val costs = results.map(x => math.pow(x._2 - x._1, 2))
    costs.foldLeft(0.0)(_ + _)
  }
}
