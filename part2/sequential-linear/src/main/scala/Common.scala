import breeze.linalg.{DenseVector, argmax, sum}
import scala.collection.mutable.ArrayBuffer

/**
  * Created by LD on 2017-07-10.
  * Common Functions and classes
  */
class DataSet(train: Seq[(DenseVector[Double], DenseVector[Double])],
              validation: Seq[(DenseVector[Double], DenseVector[Double])],
              test: Seq[(DenseVector[Double], DenseVector[Double])]) {

  val Train: Seq[(DenseVector[Double], DenseVector[Double])] = train
  val Validation: Seq[(DenseVector[Double], DenseVector[Double])] = validation
  val Test: Seq[(DenseVector[Double], DenseVector[Double])] = test
}

class CostSet(train: ArrayBuffer[(Long, Double)],
              validation: ArrayBuffer[(Long, Double)],
              test: ArrayBuffer[(Long, Double)]){

  val Train: ArrayBuffer[(Long, Double)] = train
  val Validation: ArrayBuffer[(Long, Double)] = validation
  val Test: ArrayBuffer[(Long, Double)] = test
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
    * Cost derivate of sigmoid function
    * @param output_activation
    * @param y
    * @return
    */
  def cost_derivative(output_activation: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = {
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
  def computeLoss(results: Seq[(DenseVector[Double], DenseVector[Double])]): Double = {
    val costs = results.map(x => sum((x._2 - x._1).map(y => math.pow(y, 2))))
    costs.foldLeft(0.0)(_ + _)
  }

  /**
    *
    * @param test_data
    * @param forwardFunc
    * @param startTime
    * @return
    */
  def evaluate(test_data: Seq[(DenseVector[Double], DenseVector[Double])], forwardFunc: (DenseVector[Double]) => DenseVector[Double], startTime: Long): (Int, (Long, Double)) = {
    val test_result_vector = test_data.map(x => (forwardFunc(x._1), x._2))
    val cost = (System.currentTimeMillis() - startTime, computeLoss(test_result_vector))
    val test_result = test_result_vector.map(x => (argmax(x._1), argmax(x._2)))
    (test_result.toArray.count(x => x._1 == x._2), cost)
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
                  epochCount: Int, startTime: Long, forwardFunc: (DenseVector[Double]) => DenseVector[Double]) = {
    val trainResults = evaluate(dataset.Train, forwardFunc, startTime)
    val validationResults = evaluate(dataset.Validation, forwardFunc, startTime)
    val testResults = evaluate(dataset.Test, forwardFunc, startTime)

    costs.Train += trainResults._2
    costs.Validation += validationResults._2
    costs.Test += testResults._2

    println(s"Epoch $epochCount: Train Set:     ${trainResults._1.toDouble / dataset.Train.length}")
    println(s"         Validation Set:     ${validationResults._1.toDouble / dataset.Validation.length}")
    println(s"         Test Set:     ${testResults._1.toDouble / dataset.Test.length}")
  }

  /**
    *
    * @return
    */
  def initialCosts(): CostSet = {
    val trainCosts: ArrayBuffer[(Long, Double)] = new ArrayBuffer[(Long, Double)]()

    val validationCosts: ArrayBuffer[(Long, Double)] = new ArrayBuffer[(Long, Double)]()

    val testCosts: ArrayBuffer[(Long, Double)] = new ArrayBuffer[(Long, Double)]()

    new CostSet(trainCosts, validationCosts, testCosts)
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
          updateFunc: (Seq[(DenseVector[Double], DenseVector[Double])], Double) => Unit,
          forwardFunc: (DenseVector[Double]) => DenseVector[Double]) = {
    var mini_batches: Iterator[Seq[(DenseVector[Double], DenseVector[Double])]] = Iterator.empty
    for(j <- 0 until epochs) {

      //Switch the next two lines to random shuffle the dataset before each iteration/epoch
      //mini_batches = scala.util.Random.shuffle(dataset._1).grouped(mini_batch_size)
      mini_batches = dataset.Train.grouped(mini_batch_size)

      mini_batches.foreach(x => updateFunc(x, eta))
      evaluteSets(dataset, costs, j, startTime, forwardFunc)
    }
  }
}
