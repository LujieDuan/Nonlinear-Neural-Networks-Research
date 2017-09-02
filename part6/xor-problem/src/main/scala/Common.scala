import breeze.linalg.{DenseVector, argmax, sum}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-07-10.
  * Common Functions and classes
  */
class DataSet(train: Seq[(DenseVector[Double], Double)]) {

                val Train: Seq[(DenseVector[Double], Double)] = train
              }

class CostSet(train: ArrayBuffer[(Long, Double)]){

  val Train: ArrayBuffer[(Long, Double)] = train
}


object Common {

  val Precision = 0.00000000000000000000001f

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

  /**
    * Returns the por power of num
    * @param num
    * @param por
    * @return
    */
  def powVectorVector(num: DenseVector[Double], por: DenseVector[Double]): DenseVector[Double] = {
    por.mapPairs((i, p) => pow(num(i),p))
  }

  /**
    * Returns the por power of num
    * @param num
    * @param por
    * @return
    */
  def powNumVector(num: Double, por: DenseVector[Double]): DenseVector[Double] = {
    por.mapValues(p => pow(num,p))
  }

  /**
    * Cost derivate of sigmoid function
    * @param output_activation
    * @param y
    * @return
    */
  def costDerivative(output_activation: Double, y: Double): Double = {
    output_activation - y
  }

  /**
    *
    * @param z
    * @return
    */
  def sigmoid(z: DenseVector[Double]): DenseVector[Double] = {
    z.map(x => breeze.numerics.sigmoid(x))
  }

  /**
    *
    * @param z
    * @return
    */
  def sigmoid_prime(z: DenseVector[Double]): DenseVector[Double] = {
    z.map(el => breeze.numerics.sigmoid(el) * (1 - breeze.numerics.sigmoid(el)))
  }

  /**
    *
    * @param results
    * @return
    */
  def computeLoss(results: Seq[(Double, Double)]): Double = {
    val costs = results.map(x => math.pow(x._2 - x._1, 2))
    costs.foldLeft(0.0)(_ + _)
  }

  /**
    *
    * @param test_data
    * @param forwardFunc
    * @param startTime
    * @return
    */
  def evaluate(test_data: Seq[(DenseVector[Double], Double)], forwardFunc: (DenseVector[Double]) => Double, startTime: Long): (Int, (Long, Double)) = {
    val test_result_vector = test_data.map(x => (forwardFunc(x._1), x._2))
    val cost = (System.currentTimeMillis() - startTime, computeLoss(test_result_vector))
    (test_result_vector.toArray.count(x => math.abs(x._1 - x._2) < 0.5), cost)
  }

  /**
    *
    * @param dataset
    * @param costs
    * @param epochCount
    * @param startTime
    * @param forwardFunc
    */
  def evaluteSets(dataset: DataSet, costs: CostSet,
                  epochCount: Int, startTime: Long, forwardFunc: (DenseVector[Double]) => Double): Boolean = {
    val trainResults = evaluate(dataset.Train, forwardFunc, startTime)

    costs.Train += trainResults._2

    println(s"Epoch $epochCount: Train Set: ${trainResults._1} / ${dataset.Train.length} ${ 100.0 * trainResults._1 / dataset.Train.length}%")

    trainResults._1 == dataset.Train.length
  }

  /**
    *
    * @return
    */
  def initialCosts(): CostSet = {
    val trainCosts: ArrayBuffer[(Long, Double)] = new ArrayBuffer[(Long, Double)]()

    new CostSet(trainCosts)
  }

  /**
    *
    * @param dataset
    * @param costs
    * @param epochs
    * @param mini_batch_size
    * @param eta
    * @param startTime
    * @param updateFunc
    * @param forwardFunc
    */
  def SGD(dataset: DataSet, costs: CostSet,
          epochs: Int, mini_batch_size: Int, eta: Double, startTime: Long,
          updateFunc: (Seq[(DenseVector[Double], Double)], Double) => Unit,
          forwardFunc: (DenseVector[Double]) => Double) = {
    var mini_batches: Iterator[Seq[(DenseVector[Double], Double)]] = Iterator.empty
    var j = 0
    while (j < epochs) {
      mini_batches = scala.util.Random.shuffle(dataset.Train).grouped(mini_batch_size)

      mini_batches.foreach(x => updateFunc(x, eta))

      if(j % 2000 == 0)
        if (evaluteSets(dataset, costs, j, startTime, forwardFunc))
          j = epochs

      j += 1
    }
  }
}
